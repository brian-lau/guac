# adapted from https://github.com/lisa-lab/DeepLearningTutorials

from collections import OrderedDict
import copy
import os
import re
import random
import timeit

from hyperopt import STATUS_OK

import numpy as np

import theano
from theano import tensor as T

import common
from ..util import defines
from ..util import file_handling as fh


# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(5000)


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



class RNN(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs, init_scale=0.2, initial_embeddings=None,
                 rnn_type='basic',      # 'basic', 'GRU', or 'LSTM'
                 pooling_method='max',  #'max', 'mean', 'attention1' or 'attention2',
                 add_DRLD=False,
                 bidirectional=True, bi_combine='concat'   # 'concat', 'sum', or 'mean'
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

        if add_DRLD:
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
        self.h_i_f = theano.shared(name='h_i_f', value=np.zeros(nh, dtype=theano.config.floatX))

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

        self.params = [self.emb,
                       self.W_xh, self.W_hh, self.b_h,
                       self.W_s, self.b_s,
                       self.h_i_f]
        #if add_DRLD:
        #    self.params += [self.W_drld]
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
        x = self.emb[idxs].reshape((idxs.shape[0], ne*cs))

        # create a vector for y
        y = T.ivector('y')

        def recurrence_basic(x_t, h_tm1):
            temp = T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h
            h_t = T.nnet.sigmoid(temp)
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
            h_f, _ = theano.scan(fn=recurrence_lstm, sequences=x, outputs_info=[self.h_i_f, self.c_i_f], n_steps=x.shape[0])
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_lstm_reverse, sequences=x, outputs_info=[self.h_i_r, self.c_i_r],
                                     go_backwards=True)
        else:
            h_f, _ = theano.scan(fn=recurrence_basic, sequences=x, outputs_info=[self.h_i_f], n_steps=x.shape[0])
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_basic_reverse, sequences=x, outputs_info=[self.h_i_r],
                                     go_backwards=True)

        if bidirectional:
            if bi_combine == 'max':
                h = T.maximum(h_f, h_r)
            elif bi_combine == 'mean':
                h = (h_f + h_r) / 2.0
            else:  # concatenate
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
        else:  # pooling_method == 'max'
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, nc] in R(0,1)
            p_y_given_x_sentence = T.max(s, axis=0)
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s


        # cost and gradients and learning rate
        lr = T.scalar('lr_main')
        lr_emb_fac = T.scalar('lr_emb')

        sentence_nll = T.sum(-T.log(y*p_y_given_x_sentence + (1-y)*(1-p_y_given_x_sentence)))
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, [lr_emb_fac *
                                                                            sentence_gradients[0]]
                                                                            + sentence_gradients[1:]))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        #self.get_element_weights = theano.function(inputs=[idxs], outputs=element_weights)
        self.sentence_train = theano.function(inputs=[idxs, y, lr, lr_emb_fac],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1))
                                         .dimshuffle(0, 'x')})
        if pooling_method == 'attention1' or pooling_method == 'attention2':
            self.a_sum_check = theano.function(inputs=[idxs], outputs=a_sum)

    def train(self, x, likes, y, window_size, learning_rate, emb_lr_factor):
        # concatenate words in a window
        cwords = contextwin(x, window_size)
        # make an array of these windows
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)
        # train on these sentences and normalize
        self.sentence_train(words, y, learning_rate, emb_lr_factor)
        self.normalize()

    def save(self, output_dir):
        for param in self.params:
            np.save(os.path.join(output_dir, param.name + '.npy'), param.get_value())

    def load(self, input_dir):
        for param in self.params:
            param.set_value(np.load(os.path.join(input_dir, param.name + '.npy')))

    def print_embeddings(self):
        for param in self.params:
            print param.name, param.get_value()





def main(params=None):

    if params is None:
        params = {
            'exp_name': 'test',
            'test_fold': 0,
            'n_dev_folds': 5,
            'min_doc_thresh': 1,
            'initialize_word_vectors': True,
            'vectors': 'drld_word2vec', # default_word2vec, drld_word2vec ...
            'word2vec_dim': 300,
            'add_OOV': True,
            'add_DRLD': True,
            'win': 1,                   # size of context window
            'init_scale': 0.2,
            'rnn_type': 'basic',        # basic, GRU, or LSTM
            'pooling_method': 'max',    # max, mean, or attention1/2
            'bidirectional': False,
            'bi_combine': 'concat',        # concat, max, or mean
            'nhidden': 50,             # size of hidden units
            'lr': 0.1,                  # learning rate
            'lr_emb_fac': 0.2,            # factor to modify learning rate for embeddings
            'decay_delay': 5,           # number of epochs with no improvement before decreasing learning rate
            'decay_factor': 0.5,        # factor by which to multiply learning rate in case of delay
            'n_epochs': 20,
            'save_model': True,
            'seed': 42,
            'verbose': 1,
            'reuse': False,
            'T_orig': 0.04,
            'tau': 0.01
        }
    print params

    # seed the random number generators
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']

    np.random.seed(params['seed'])
    random.seed(params['seed'])

    for dev_fold in range(params['n_dev_folds']):
        print "dev fold =", dev_fold

        output_dir = fh.makedirs(defines.exp_dir, 'rnn', params['exp_name'], 'fold' + str(dev_fold))

        all_data, words2idx, items, all_labels = common.load_data(datasets, params['test_fold'], dev_fold,
                                                                  params['min_doc_thresh'])
        train_xy, valid_xy, test_xy = all_data
        train_lex, train_y = train_xy
        valid_lex, valid_y = valid_xy
        test_lex, test_y = test_xy
        train_items, dev_items, test_items = items
        vocsize = len(words2idx.keys())
        idx2words = dict((k, v) for v, k in words2idx.iteritems())

        n_sentences = len(train_lex)
        print "vocsize = ", vocsize, 'n_train', n_sentences

        n_items, n_codes = all_labels.shape

        # get the words in the sentences for the test and validation sets
        words_valid = [map(lambda x: idx2words[x], w) for w in valid_lex]
        groundtruth_test = test_y[:]
        words_test = [map(lambda x: idx2words[x], w) for w in test_lex]

        initial_embeddings = common.load_embeddings(params, words2idx)
        emb_dim = initial_embeddings.shape[0]

        best_valid_f1s = []
        best_test_f1s = []

        print "Building RNN"
        rnn = RNN(nh=params['nhidden'],
                  nc=n_codes,
                  ne=vocsize,
                  de=emb_dim,
                  cs=params['win'],
                  add_DRLD=params['add_DRLD'],
                  initial_embeddings=initial_embeddings,
                  init_scale=params['init_scale'],
                  rnn_type=params['rnn_type'],
                  pooling_method=params['pooling_method'],
                  bidirectional=params['bidirectional'],
                  bi_combine=params['bi_combine']
                  )

        # train with early stopping on validation set
        best_f1 = -np.inf
        params['clr'] = params['lr']
        for e in xrange(params['n_epochs']):
            # shuffle

            shuffle([train_lex, train_y, train_items], params['seed'])   # shuffle the input data
            params['ce'] = e                # store the current epoch
            tic = timeit.default_timer()

            for i, (x, y) in enumerate(zip(train_lex, train_y)):
                if re.search('Likes', train_items[i]) is not None:
                    likes = 1
                else:
                    likes = 0
                rnn.train(x, likes, y, params['win'], params['clr'], params['lr_emb_fac'])
                print '[learning] epoch %i >> %2.2f%%' % (
                    e, (i + 1) * 100. / float(n_sentences)),
                print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic),
                sys.stdout.flush()

            # evaluation // back into the real world : idx -> words
            print ""

            print rnn.classify(np.asarray(contextwin(train_lex[0], params['win'])).astype('int32'))
            #print rnn.get_element_weights(np.asarray(contextwin(train_lex[0], params['win'])).astype('int32'))
            if params['pooling_method'] == 'attention1' or params['pooling_method'] == 'attention2':
                print rnn.a_sum_check(np.asarray(contextwin(train_lex[0], params['win'])).astype('int32'))

            """
            predictions_train = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                                 for x in train_lex]
            predictions_test = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                                for x in test_lex]
            predictions_valid = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                                 for x in valid_lex]
            """

            predictions_train = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')) for x in train_lex]
            predictions_test = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')) for x in test_lex]
            predictions_valid = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')) for x in valid_lex]

            train_f1 = common.calc_mean_f1(predictions_train, train_y)
            test_f1 = common.calc_mean_f1(predictions_test, test_y)
            valid_f1 = common.calc_mean_f1(predictions_valid, valid_y)

            question_f1s = []
            question_pps = []

            print "train_f1 =", train_f1, "valid_f1 =", valid_f1, "test_f1 =", test_f1

            if valid_f1 > best_f1:
                best_rnn = copy.deepcopy(rnn)
                best_f1 = valid_f1

                if params['verbose']:
                    print('NEW BEST: epoch', e,
                          'valid f1', valid_f1,
                          'best test f1', test_f1)

                params['tr_f1'] = train_f1
                params['te_f1'] = test_f1
                params['v_f1'] = valid_f1
                params['be'] = e            # store the current epoch as a new best

            # learning rate decay if no improvement in a given number of epochs
            if abs(params['be']-params['ce']) >= params['decay_delay']:
                params['clr'] *= params['decay_factor']
                print "Reverting to current best; new learning rate = ", params['clr']
                # also reset to the previous best
                rnn = best_rnn

            # also reduce learning rate if we can't even fit the training data
            if train_f1 == 0.0:
                params['clr'] *= params['decay_factor']

            if params['clr'] < 1e-5:
                break

            if best_f1 == 1.0:
                break

        best_rnn.print_embeddings()

        if params['save_model']:
            predictions_valid = [best_rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')) for x in valid_lex]
            best_rnn.save(output_dir)
            common.write_predictions(datasets, predictions_valid, dev_items, output_dir)

        print('BEST RESULT: epoch', params['be'],
              'train F1 ', params['tr_f1'],
              'valid F1', params['v_f1'],
              'best test F1', params['te_f1'],
              'with the model', output_dir)

        best_valid_f1s.append(params['v_f1'])
        best_test_f1s.append(params['te_f1'])

    return {'loss': -np.max(best_valid_f1s),
            'mean_valid_f1': np.mean(best_valid_f1s),
            'valid_f1s': best_valid_f1s,
            'test_f1s': best_test_f1s,
            'status': STATUS_OK
            }


if __name__ == '__main__':
    main()