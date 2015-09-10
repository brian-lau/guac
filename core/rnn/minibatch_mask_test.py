# adapted from https://github.com/lisa-lab/DeepLearningTutorials

from collections import OrderedDict
import copy
import os
import re
import random
import timeit

from hyperopt import STATUS_OK

import numpy as np
import pandas as pd
from scipy import stats


import theano
from theano import tensor as T

import common
from ..util import defines
from ..util import file_handling as fh
from ..experiment import reusable_holdout
from ..experiment import evaluation


# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(5000)

THEANO_FLAGS='floatX=float32'

# utils functions
def shuffle(lol, seed=None):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def generate_data(n, nVocab=2, seqMin=3, seqMax=10):
    x = []
    y = []
    for i in range(n):
        seqLen = np.random.randint(seqMin, seqMax)
        seq = list(np.random.randint(0, nVocab, seqLen))
        x.append(seq)
        pairs = [[seq[i], seq[i+1]] for (i, v) in enumerate(seq[:-1])]
        if [1, 1] in pairs:
            y.append([1])
        else:
            y.append([0])

    return x, y


class RNN(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs, init_scale=0.2, initial_embeddings=None,
                 rnn_type='basic',      # 'basic', 'GRU', or 'LSTM'
                 pooling_method='max',  #'max', 'mean', 'attention1' or 'attention2',
                 extra_input_dims=0, train_embeddings=False,
                 bidirectional=False, bi_combine='concat'   # 'concat', 'sum', or 'mean'
                ):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''

        # initialize parameters
        dx = de * cs
        if extra_input_dims > 0:
            dx += extra_input_dims
        bi = 1
        if bidirectional and bi_combine == 'concat':
            bi = 2
        if initial_embeddings is None:
            self.emb = theano.shared(name='embeddings',
                                     value=init_scale * np.random.uniform(-1.0, 1.0,
                                                                          (ne, de)).astype(theano.config.floatX))
                                                                    #(ne+1, de)) # add one for padding at the end
        else:
            self.emb = theano.shared(name='embeddings', value=initial_embeddings.astype(theano.config.floatX))

        if extra_input_dims > 0:
            self.W_drld = theano.shared(name='W_drld', value=init_scale * np.random.uniform(-1.0, 1.0, (1, nh))
                                        .astype(theano.config.floatX))
        # common paramters (feeding into hidden node)
        self.W_xh = theano.shared(name='W_xh', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                  .astype(theano.config.floatX))
        self.W_hh = theano.shared(name='W_hh', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                  .astype(theano.config.floatX))
        self.b_h = theano.shared(name='b_h', value=np.array(np.random.uniform(0.0, 1.0, nh),
                                                            dtype=theano.config.floatX))

        # output layer parameters
        self.W_s = theano.shared(name='W_s', value=init_scale * np.random.uniform(-1.0, 1.0, (nh * bi, nc))
                                 .astype(theano.config.floatX))
        self.b_s = theano.shared(name='b_s', value=np.zeros(nc, dtype=theano.config.floatX))

        # temporary parameters
        #self.h_i_f = theano.shared(name='h_i_f', value=np.zeros(nh, dtype=theano.config.floatX))

        if bidirectional:
            self.h_i_r = theano.shared(name='h_i_r', value=np.zeros(nh, dtype=theano.config.floatX))

        # Attention parameters
        if pooling_method == 'attention1' or pooling_method == 'attention2':
            self.W_a = theano.shared(name='W_a', value=init_scale * np.random.uniform(-1.0, 1.0, (bi*nh, 1))
                                     .astype(theano.config.floatX))
            self.b_a = theano.shared(name='b_a', value=0.0)

        # GRU parameters
        if rnn_type == 'GRU':
            self.W_xr = theano.shared(name='W_xr', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            self.W_hr = theano.shared(name='W_hr', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_r = theano.shared(name='b_r', value=np.zeros(nh, dtype=theano.config.floatX))
            self.W_xz = theano.shared(name='W_xz', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            self.W_hz = theano.shared(name='W_hz', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_z = theano.shared(name='b_z', value=np.zeros(nh, dtype=theano.config.floatX))

        # LSTM paramters
        if rnn_type == 'LSTM':
            # forget gate (needs special initialization)
            self.W_xf = theano.shared(name='W_xf', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            self.W_hf = theano.shared(name='W_hf', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_cf = theano.shared(name='W_cf', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_f = theano.shared(name='b_f', value=np.array(np.random.uniform(0.0, 1.0, nh),
                                                                dtype=theano.config.floatX))
            # input gate
            self.W_xi = theano.shared(name='W_xi', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            self.W_hi = theano.shared(name='W_hi', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_ci = theano.shared(name='W_ci', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_i = theano.shared(name='b_i', value=np.zeros(nh, dtype=theano.config.floatX))

            # output gate
            self.W_xo = theano.shared(name='W_xo', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            self.W_ho = theano.shared(name='W_ho', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_co = theano.shared(name='W_co', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_o = theano.shared(name='b_o', value=np.zeros(nh, dtype=theano.config.floatX))

            # use normal ->hidden weights for memory cell

            # temp
            self.c_i_f = theano.shared(name='c_i_f', value=np.zeros(nh, dtype=theano.config.floatX))
            if bidirectional:
                self.c_i_r = theano.shared(name='c_i_r', value=np.zeros(nh, dtype=theano.config.floatX))

        self.params = [self.W_xh, self.W_hh, self.b_h,
                       self.W_s, self.b_s]
        if train_embeddings:
            self.params += [self.emb]
        if pooling_method == 'attention':
            self.params += [self.W_a, self.b_a]
        if rnn_type == 'GRU':
            self.params += [self.W_xr, self.W_hr, self.b_r,
                            self.W_xz, self.W_hz, self.b_z]
        if rnn_type == 'LSTM':
            self.params += [self.W_xf, self.W_hf, self.W_cf, self.b_f,
                            self.W_xi, self.W_hi, self.W_ci, self.b_i,
                            self.W_xo, self.W_ho, self.W_co, self.b_o,
                            self.c_i_f]
            if bidirectional:
                self.params += [self.c_i_r]
        if bidirectional:
            self.params += [self.h_i_r]

        # create an X object based on the size of the object at the index [elements, emb_dim * window]
        idxs = T.imatrix()
        if extra_input_dims:
            extra = T.imatrix()
            x = T.concatenate([self.emb[idxs].reshape((idxs.shape[0], de*cs)),
                               T.repeat(extra, idxs.shape[0], axis=0)], axis=1)
        else:
            x = self.emb[idxs].reshape((idxs.shape[0], de*cs))

        # create a vector for y
        y = T.ivector('y')

        def recurrence_basic(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h)
            return h_t

        def recurrence_basic_reverse(x_t, h_tp1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_xh) + T.dot(h_tp1, self.W_hh) + self.b_h)
            return h_t

        def recurrence_gru(x_t, h_tm1):
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr) + self.b_r)
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz) + self.b_z)
            g_t = T.tanh(T.dot(x_t, self.W_xh) + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)
            h_t = (1 - z_t) * h_tm1 + z_t * g_t
            return h_t

        def recurrence_gru_reverse(x_t, h_tp1):
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tp1, self.W_hr) + self.b_r)
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tp1, self.W_hz) + self.b_z)
            g_t = T.tanh(T.dot(x_t, self.W_xh) + r_t * T.dot(h_tp1, self.W_hh) + self.b_h)
            h_t = (1 - z_t) * h_tp1 + z_t * g_t
            return h_t

        def recurrence_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            d_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h)
            c_t = f_t * c_tm1 + i_t * d_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * c_t
            return [h_t, c_t]

        def recurrence_lstm_reverse(x_t, h_tp1, c_tp1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tp1, self.W_hi) + T.dot(c_tp1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tp1, self.W_hf) + T.dot(c_tp1, self.W_cf) + self.b_f)
            d_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot(h_tp1, self.W_hh) + self.b_h)
            c_t = f_t * c_tp1 + i_t * d_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tp1, self.W_ho) + T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * c_t
            return [h_t, c_t]

        h_r = None

        if rnn_type == 'GRU':
            h_f, _ = theano.scan(fn=recurrence_gru, sequences=x, outputs_info=[self.h_i_f], n_steps=x.shape[0])
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_gru_reverse, sequences=x, outputs_info=[self.h_i_r],
                                     go_backwards=True)
        elif rnn_type == 'LSTM':
            [h_f, c_f], _ = theano.scan(fn=recurrence_lstm, sequences=x,
                                        outputs_info=[self.h_i_f, self.c_i_f], n_steps=x.shape[0])
            if bidirectional:
                [h_r, c_r], _ = theano.scan(fn=recurrence_lstm_reverse, sequences=x,
                                            outputs_info=[self.h_i_r, self.c_i_r], go_backwards=True)
        else:
            h_f, _ = theano.scan(fn=recurrence_basic, sequences=x,
                                 outputs_info=[T.alloc(np.asarray(0., dtype=T.config.floatX), nh)],
                                 n_steps=x.shape[0])
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_basic_reverse, sequences=x, outputs_info=[self.h_i_r],
                                     go_backwards=True)

        if bidirectional:
            # reverse the second hidden layer so it lines up with the first
            h_r = h_r[::-1, :]
            if bi_combine == 'max':
                h = T.maximum(h_f, h_r)
            elif bi_combine == 'mean':
                h = (h_f + h_r) / 2.0
            else:  # concatenate
                #h = theano.printing.Print('h:')(T.concatenate([h_fp, h_rp], axis=1))
                h = T.concatenate([h_f, h_r], axis=1)
        else:
            h = h_f

        a_sum = T.sum([1])
        if pooling_method == 'attention1':  # combine hidden nodes, then transform and sigmoid
            # SOFTMAX normalizes across the row (axis=1)
            a = T.nnet.softmax((T.dot(h, self.W_a) + self.b_a).T)  # [1, n_elements]: normalized vector
            a_sum = T.sum(a)    # to check a is normalized
            p_y_given_x_sentence = T.nnet.sigmoid(T.dot(T.dot(a, h), self.W_s) + self.b_s)  # [1, nc] in R(0,1)
            y_pred = T.max(p_y_given_x_sentence, axis=0) > 0.5  # note, max is just to coerce into proper shape
            element_weights = T.outer(a, p_y_given_x_sentence)  # [ne, nc]
        elif pooling_method == 'attention2': # transform hidden nodes, sigmoid, then combine
            a = T.nnet.softmax((T.dot(h, self.W_a) + self.b_a).T)  # [1, n_elements]: normalized vector
            a_sum = T.sum(a)
            temp = T.nnet.sigmoid(T.dot(h, self.W_s) + self.b_s)  # [ne x nc]
            p_y_given_x_sentence = T.dot(a, temp)  # [1, nc] in R(0,1)
            y_pred = T.max(p_y_given_x_sentence, axis=0) > 0.5  # note, max is just to coerce into proper shape
            element_weights = T.repeat(a.T, nc, axis=1) * temp   # [ne, nc]
        elif pooling_method == 'mean':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, nc] in R(0,1)
            p_y_given_x_sentence = T.mean(s, axis=0)
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        elif pooling_method == 'max':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, nc] in R(0,1)
            p_y_given_x_sentence = T.max(s, axis=0)
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        elif pooling_method == 'last':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, nc] in R(0,1)
            p_y_given_x_sentence = s[-1, :]
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        else:
            sys.exit("Pooling method not recognized")


        # cost and gradients and learning rate
        lr = T.scalar('lr_main')
        lr_emb_fac = T.scalar('lr_emb')

        sentence_nll = T.sum(-T.log(y*p_y_given_x_sentence + (1-y)*(1-p_y_given_x_sentence)))
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, [lr_emb_fac *
                                                                            sentence_gradients[0]]
                                                                            + sentence_gradients[1:]))

        self.print_gradients = theano.function(inputs=[idxs, y], outputs=sentence_gradients)

        # theano functions to compile
        if extra_input_dims > 0:
            self.sentence_classify = theano.function(inputs=[idxs, extra], outputs=y_pred)
            self.sentence_train = theano.function(inputs=[idxs, extra, y, lr, lr_emb_fac],
                                                       outputs=sentence_nll,
                                                       updates=sentence_updates)
            if pooling_method == 'attention1' or pooling_method == 'attention2':
                self.a_sum_check = theano.function(inputs=[idxs, extra], outputs=a_sum)
        else:
            self.sentence_classify = theano.function(inputs=[idxs], outputs=y_pred)
            self.sentence_train = theano.function(inputs=[idxs, y, lr, lr_emb_fac],
                                                  outputs=sentence_nll,
                                                  updates=sentence_updates)
            if pooling_method == 'attention1' or pooling_method == 'attention2':
                self.a_sum_check = theano.function(inputs=[idxs], outputs=a_sum)

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1))
                                         .dimshuffle(0, 'x')})

    def classify(self, x, window_size, extra_input_dims=0, extra=None):

        cwords = contextwin(x, window_size)
        # make an array of these windows
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)

        if extra_input_dims > 0:
            extra = np.array(extra).astype('int32').reshape((1, extra_input_dims))
            return self.sentence_classify(words, extra)
        else:
            return self.sentence_classify(words)

    def train(self, x, y, window_size, learning_rate, emb_lr_factor, extra_input_dims=0, extra=None):
        # concatenate words in a window
        cwords = contextwin(x, window_size)
        # make an array of these windows
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)

        # train on these sentences and normalize
        if extra_input_dims > 0:
            extra = np.array(extra).astype('int32').reshape((1, extra_input_dims))
            self.sentence_train(words, extra, y, learning_rate, emb_lr_factor)
        else:
            self.sentence_train(words, y, learning_rate, emb_lr_factor)
        self.normalize()

    def save(self, output_dir):
        for param in self.params:
            np.save(os.path.join(output_dir, param.name + '.npy'), param.get_value())

    def load(self, input_dir):
        for param in self.params:
            param.set_value(np.load(os.path.join(input_dir, param.name + '.npy')))

    def print_params(self):
        for param in self.params:
            print param.name, param.get_value()


    def print_all_gradients(self, x, y):
        cwords = contextwin(x, 1)
        # make an array of these windows
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)
        print self.print_gradients(words, y)



def main(params=None):

    if params is None:
        params = {
            'exp_name': 'ensemble_test',
            'test_fold': 0,
            'n_dev_folds': 1,
            'min_doc_thresh': 1,
            'initialize_word_vectors': True,
            'vectors': 'anes_word2vec',  # default_word2vec, anes_word2vec ...
            'word2vec_dim': 300,
            'init_scale': 0.2,
            'add_OOV': True,
            'win': 1,                   # size of context window
            'add_DRLD': False,
            'rnn_type': 'basic',        # basic, GRU, or LSTM
            'n_hidden': 2,             # size of hidden units
            'pooling_method': 'max',    # max, mean, last, or attention1/2
            'bidirectional': False,
            'bi_combine': 'mean',        # concat, max, or mean
            'train_embeddings': False,
            'lr': 0.01,                  # learning rate
            'lr_emb_fac': 0.2,            # factor to modify learning rate for embeddings
            'decay_delay': 5,           # number of epochs with no improvement before decreasing learning rate
            'decay_factor': 0.5,        # factor by which to multiply learning rate in case of delay
            'n_epochs': 5,
            'add_OOV_noise': False,
            'OOV_noise_prob': 0.01,
            'ensemble': False,
            'save_model': True,
            'seed': 42,
            'verbose': 1,
            'reuse': False,
            'orig_T': 0.04,
            'tau': 0.01
        }

    reuser = None
    if params['reuse']:
        reuser = reusable_holdout.ReuseableHoldout(T=params['orig_T'], tau=params['tau'])


    n_dev_folds = 1

    for dev_fold in range(params['n_dev_folds']):
        print "dev fold =", dev_fold

        #output_dir = fh.makedirs(defines.exp_dir, 'rnn', params['exp_name'], 'fold' + str(dev_fold))

        #all_data, words2idx, items, all_labels = common.load_data(datasets, params['test_fold'], dev_fold,
        #                                                          params['min_doc_thresh'])
        #train_xy, valid_xy, test_xy = all_data
        #train_lex, train_y = train_xy
        #valid_lex, valid_y = valid_xy
        #test_lex, test_y = test_xy
        #train_items, dev_items, test_items = items
        #vocsize = len(words2idx.keys())
        #idx2words = dict((k, v) for v, k in words2idx.iteritems())
        #best_test_predictions = None

        extra_input_dims = 0

        vocsize = 2
        seq_min = 3
        seq_max = 5
        train_lex, train_y = generate_data(60, vocsize, seq_min, seq_max)
        valid_lex, valid_y = generate_data(60, vocsize, seq_min, seq_max)
        test_lex, test_y = generate_data(60, vocsize, seq_min, seq_max)

        #init_embed = np.reshape(np.array([0, 1]), [2, 1])
        init_embed = np.eye(2)
        vs, de = init_embed.shape
        assert vs == vocsize

        n_sentences = len(train_lex)
        print "vocsize = ", vocsize, 'n_train', n_sentences


        print "Building RNN"
        rnn = RNN(nh=params['n_hidden'],
                  nc=1,
                  ne=vocsize,
                  de=de,
                  cs=params['win'],
                  initial_embeddings=init_embed,
                  init_scale=params['init_scale'],
                  rnn_type=params['rnn_type'],
                  train_embeddings=params['train_embeddings'],
                  pooling_method=params['pooling_method'],
                  bidirectional=params['bidirectional'],
                  bi_combine=params['bi_combine']
                  )

        # train with early stopping on validation set
        best_f1 = -np.inf
        params['clr'] = params['lr']
        for e in xrange(params['n_epochs']):
            # shuffle
            shuffle([train_lex, train_y], params['seed'])   # shuffle the input data
            params['ce'] = e                # store the current epoch
            tic = timeit.default_timer()
            test_f1 = 0
            valid_f1 = 0

            print "epoch = ", e

            for i, x in enumerate(train_lex):
                n_words = len(x)
                y = train_y[i]
                y_hat_pre = rnn.classify(x, params['win'])

                #rnn.print_params()

                #print x, y
                #rnn.print_all_gradients(x, y)

                rnn.train(x, y, params['win'], params['clr'], params['lr_emb_fac'])
                #print '[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / float(n_sentences))
                #print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic)

                y_hat_post = rnn.classify(x, params['win'])
                predictions_test = [rnn.classify(t, params['win'],
                                                 extra_input_dims, None) for t in test_lex]
                predictions_valid = [rnn.classify(t, params['win'],
                                                  extra_input_dims, None) for t in valid_lex]

                test_f1 = common.calc_mean_f1(predictions_test, test_y)
                valid_f1 = common.calc_mean_f1(predictions_valid, valid_y)

                #sys.stdout.flush()
                print x, y, y_hat_pre, y_hat_post, test_f1, valid_f1


            # evaluation // back into the real world : idx -> words
            predictions_train = [rnn.classify(x, params['win'],
                                              extra_input_dims, None) for i, x in enumerate(train_lex)]
            predictions_test = [rnn.classify(x, params['win'],
                                             extra_input_dims, None) for i, x in enumerate(test_lex)]
            predictions_valid = [rnn.classify(x, params['win'],
                                              extra_input_dims, None) for i, x in enumerate(valid_lex)]

            train_f1 = common.calc_mean_f1(predictions_train, train_y)
            test_f1 = common.calc_mean_f1(predictions_test, test_y)
            valid_f1 = common.calc_mean_f1(predictions_valid, valid_y)

            if reuser is not None:
                valid_f1 = reuser.mask_value(valid_f1, train_f1)

            question_f1s = []
            question_pps = []

            print "train_f1 =", train_f1, "valid_f1 =", valid_f1, "test_f1 =", test_f1

            if valid_f1 > best_f1:
                #best_rnn = copy.deepcopy(rnn)
                best_f1 = valid_f1
                best_test_predictions = predictions_test

                if params['verbose']:
                    print('NEW BEST: epoch', e,
                          'valid f1', valid_f1,
                          'best test f1', test_f1)

                params['tr_f1'] = train_f1
                params['te_f1'] = test_f1
                params['v_f1'] = valid_f1
                params['be'] = e            # store the current epoch as a new best


            if best_f1 == 1.0:
                break

        print('BEST RESULT: epoch', params['be'],
              'train F1 ', params['tr_f1'],
              'valid F1', params['v_f1'],
              'best test F1', params['te_f1'])

        #best_rnn.print_params()
        rnn.print_params()


if __name__ == '__main__':
    report = main()
    print report