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




class RNN(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs, init_scale=0.2, initial_embeddings=None,
                 rnn_type='basic',      # 'basic', 'GRU', or 'LSTM'
                 pooling_method='max',  #'max', 'mean', 'attention1' or 'attention2',
                 extra_input_dims=0, train_embeddings=True,
                 bidirectional=True, bi_combine='concat',   # 'concat', 'sum', or 'mean'
                 xavier_init=False
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

        # common paramters (feeding into hidden node)
        if xavier_init:
            init_scale = np.sqrt(6/float(dx+nh))
        self.W_xh = theano.shared(name='W_xh', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                  .astype(theano.config.floatX))
        if xavier_init:
            init_scale = np.sqrt(6/float(nh+nh))
        self.W_hh = theano.shared(name='W_hh', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                  .astype(theano.config.floatX))
        #self.b_h = theano.shared(name='b_h', value=np.array(np.random.uniform(0.0, 1.0, nh),
        #                                                    dtype=theano.config.floatX))
        self.b_h = theano.shared(name='b_h', value=np.zeros(nh, dtype=theano.config.floatX))


        # output layer parameters
        if xavier_init:
            init_scale = np.sqrt(6/float(nh*bi+nc))
        self.W_s = theano.shared(name='W_s', value=init_scale * np.random.uniform(-1.0, 1.0, (nh * bi, nc))
                                 .astype(theano.config.floatX))
        self.b_s = theano.shared(name='b_s', value=np.zeros(nc, dtype=theano.config.floatX))

        # temporary parameters
        #self.h_i_f = theano.shared(name='h_i_f', value=np.zeros((2, nh), dtype=theano.config.floatX))

        #if bidirectional:
        #    self.h_i_r = theano.shared(name='h_i_r', value=np.zeros(nh, dtype=theano.config.floatX))

        # Attention parameters
        if pooling_method == 'attention1' or pooling_method == 'attention2':
            if xavier_init:
                init_scale = np.sqrt(6/float(bi+nh))
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
            if xavier_init:
                init_scale = np.sqrt(6/float(dx+nh))
            self.W_xf = theano.shared(name='W_xf', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            if xavier_init:
                init_scale = np.sqrt(6/float(nh+nh))
            self.W_hf = theano.shared(name='W_hf', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_cf = theano.shared(name='W_cf', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_f = theano.shared(name='b_f', value=np.array(np.random.uniform(0.8, 1.0, nh),
                                                                dtype=theano.config.floatX))
            # input gate
            if xavier_init:
                init_scale = np.sqrt(6/float(dx+nh))
            self.W_xi = theano.shared(name='W_xi', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            if xavier_init:
                init_scale = np.sqrt(6/float(nh+nh))
            self.W_hi = theano.shared(name='W_hi', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_ci = theano.shared(name='W_ci', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_i = theano.shared(name='b_i', value=np.zeros(nh, dtype=theano.config.floatX))

            # output gate
            if xavier_init:
                init_scale = np.sqrt(6/float(dx+nh))
            self.W_xo = theano.shared(name='W_xo', value=init_scale * np.random.uniform(-1.0, 1.0, (dx, nh))
                                      .astype(theano.config.floatX))
            if xavier_init:
                init_scale = np.sqrt(6/float(nh+nh))
            self.W_ho = theano.shared(name='W_ho', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.W_co = theano.shared(name='W_co', value=init_scale * np.random.uniform(-1.0, 1.0, (nh, nh))
                                      .astype(theano.config.floatX))
            self.b_o = theano.shared(name='b_o', value=np.zeros(nh, dtype=theano.config.floatX))

            # use normal ->hidden weights for memory cell

            # temp
            #self.c_i_f = theano.shared(name='c_i_f', value=np.zeros(nh, dtype=theano.config.floatX))
            #if bidirectional:
            #    self.c_i_r = theano.shared(name='c_i_r', value=np.zeros(nh, dtype=theano.config.floatX))

        self.params = [self.W_xh, self.W_hh, self.b_h,
                       self.W_s]
        self.params += [self.b_s]
        #self.params += [self.h_i_f]
        if train_embeddings:
            self.params += [self.emb]
        if pooling_method == 'attention1' or pooling_method == 'attention2':
            self.params += [self.W_a, self.b_a]
        if rnn_type == 'GRU':
            self.params += [self.W_xr, self.W_hr, self.b_r,
                            self.W_xz, self.W_hz, self.b_z]
        if rnn_type == 'LSTM':
            self.params += [self.W_xf, self.W_hf, self.W_cf, self.b_f,
                            self.W_xi, self.W_hi, self.W_ci, self.b_i,
                            self.W_xo, self.W_ho, self.W_co, self.b_o]
                            #self.c_i_f]
            #if bidirectional:
            #    self.params += [self.c_i_r]
        #if bidirectional:
        #    self.params += [self.h_i_r]

        # create an X object based on the size of the object at the index [elements, emb_dim * window]
        idxs = T.tensor3('idxs', dtype='int32')
        if extra_input_dims:
            extra = T.tensor3('extra')
            extra_3d = extra.repeat(idxs.shape[0], axis=0)
            #x = T.concatenate([self.emb[idxs].reshape((idxs.shape[0], de*cs)),
            #                   T.repeat(extra, idxs.shape[0], axis=0)], axis=1)
            #temp = T.printing.Print('temp')(self.emb[idxs].reshape((idxs.shape[0], idxs.shape[1], de*cs)))
            temp = self.emb[idxs].reshape((idxs.shape[0], idxs.shape[1], de*cs))
            x = T.concatenate([temp, extra_3d], axis=2)
        else:
            #x = T.printing.Print('x')(self.emb[idxs])
            x = self.emb[idxs].reshape((idxs.shape[0], idxs.shape[1], de*cs))  # [n_elements, minibatch_size, emb_dim]
            #x = self.emb[idxs]

        y = T.imatrix('y')
        mask = T.tensor3('mask')
        mask_3d = mask.repeat(nh, axis=2)
        minibatch_size = T.iscalar()

        def recurrence_basic(x_t, mask_t, h_tm1):
            #h_t = theano.printing.Print('h_t')(T.nnet.sigmoid(T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h))
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h)
            #masked_h_t = T.printing.Print('masked_h_t')(mask_t * h_t + (1 - mask_t) * h_tm1)
            # apply the mask to propogate the last (unmaksed) element in sequence to the end
            return mask_t * h_t + (1 - mask_t) * h_tm1
            #return h_t

        def recurrence_basic_reverse(x_t, mask_t, h_tp1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_xh) + T.dot(h_tp1, self.W_hh) + self.b_h)
            return mask_t * h_t + (1 - mask_t) * h_tp1

        def recurrence_gru(x_t, mask_t, h_tm1):
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr) + self.b_r)
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz) + self.b_z)
            g_t = T.tanh(T.dot(x_t, self.W_xh) + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)
            h_t = (1 - z_t) * h_tm1 + z_t * g_t
            return mask_t * h_t + (1 - mask_t) * h_tm1

        def recurrence_gru_reverse(x_t, mask_t, h_tp1):
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tp1, self.W_hr) + self.b_r)
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tp1, self.W_hz) + self.b_z)
            g_t = T.tanh(T.dot(x_t, self.W_xh) + r_t * T.dot(h_tp1, self.W_hh) + self.b_h)
            h_t = (1 - z_t) * h_tp1 + z_t * g_t
            return mask_t * h_t + (1 - mask_t) * h_tp1

        def recurrence_lstm(x_t, mask_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            d_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot(h_tm1, self.W_hh) + self.b_h)
            c_t = f_t * c_tm1 + i_t * d_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * c_t
            return [mask_t * h_t + (1 - mask_t) * h_tm1, mask_t * c_t + (1 - mask_t) * c_tm1]

        def recurrence_lstm_reverse(x_t, mask_t, h_tp1, c_tp1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tp1, self.W_hi) + T.dot(c_tp1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tp1, self.W_hf) + T.dot(c_tp1, self.W_cf) + self.b_f)
            d_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot(h_tp1, self.W_hh) + self.b_h)
            c_t = f_t * c_tp1 + i_t * d_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tp1, self.W_ho) + T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * c_t
            return [mask_t * h_t + (1 - mask_t) * h_tp1, mask_t * c_t + (1 - mask_t) * c_tp1]

        h_r = None

        if rnn_type == 'GRU':
            h_f, _ = theano.scan(fn=recurrence_gru, sequences=[x, mask_3d],
                                 outputs_info=[T.alloc(np.array(0.), minibatch_size, nh)],
                                 n_steps=x.shape[0])
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_gru_reverse, sequences=[x, mask_3d],
                                     outputs_info=[T.alloc(np.array(0.), minibatch_size, nh)],
                                     go_backwards=True)
        elif rnn_type == 'LSTM':
            [h_f, c_f], _ = theano.scan(fn=recurrence_lstm, sequences=[x, mask_3d],
                                        outputs_info=[T.alloc(np.array(0.), minibatch_size, nh),
                                                      T.alloc(np.array(0.), minibatch_size, nh)],
                                        n_steps=x.shape[0])
            if bidirectional:
                [h_r, c_r], _ = theano.scan(fn=recurrence_lstm_reverse, sequences=[x, mask_3d],
                                            outputs_info=[T.alloc(np.array(0.), minibatch_size, nh),
                                                          T.alloc(np.array(0.), minibatch_size, nh)],
                                            go_backwards=True)
                #[h_r, c_r], _ = theano.scan(fn=recurrence_lstm_reverse, sequences=x,
                #                            outputs_info=[self.h_i_r, self.c_i_r], go_backwards=True)
        else:
            #h_f, _ = theano.scan(fn=recurrence_basic, sequences=x, outputs_info=[self.h_i_f], n_steps=x.shape[0])
            temp, _ = theano.scan(fn=recurrence_basic, sequences=[x, mask_3d],
                                 outputs_info=[T.alloc(np.array(0.), minibatch_size, nh)],
                                 n_steps=x.shape[0])
            #h_f = theano.printing.Print('h_f')(temp)
            h_f = temp
            if bidirectional:
                h_r, _ = theano.scan(fn=recurrence_basic_reverse, sequences=[x, mask_3d],
                                     outputs_info=[T.alloc(np.array(0.), minibatch_size, nh)],
                                     go_backwards=True)

        if bidirectional:
            # reverse the second hidden layer so it lines up with the first
            h_r = h_r[::-1, :, :]
            if bi_combine == 'max':
                h = T.maximum(h_f, h_r)
            elif bi_combine == 'mean':
                h = (h_f + h_r) / 2.0
            else:  # concatenate
                #h = theano.printing.Print('h:')(T.concatenate([h_fp, h_rp], axis=1))
                h = T.concatenate([h_f, h_r], axis=2)
        else:
            #temp = T.printing.Print('isnan')(T.max(T.isnan(h_f)))
            #h = h_f * (1-temp)
            h = h_f  #[n_elements, minibatch_size, n_hidden] (?)

        a_sum = T.sum([1])
        if pooling_method == 'attention1':  # combine hidden nodes, then transform and sigmoid
            # THIS IS NOT WORKIGN...

            # SOFTMAX normalizes across the row (axis=1)
            #a = T.nnet.softmax((T.dot(h, self.W_a) + self.b_a).T)

            temp = T.dot(h, self.W_a) + self.b_a
            # softmax?
            a = T.exp(temp)/T.exp(temp).sum(axis=0, keepdims=True)

            a_sum = T.sum(a, )    # to check a is normalized
            a_rep = T.repeat(a, nh*bi, axis=2)
            weighted_sum = T.sum(h * a_rep, axis=0)

            p_y_given_x_sentence = T.nnet.sigmoid(T.dot(weighted_sum, self.W_s) + self.b_s)  # [1, nc] in R(0,1)
            y_pred = p_y_given_x_sentence  > 0.5  # note, max is just to coerce into proper shape
            #element_weights = T.outer(a, p_y_given_x_sentence)  # [ne, nc]

            #p_y_given_x_sentence = T.nnet.sigmoid(T.dot(T.dot(a, h), self.W_s) + self.b_s)  # [1, nc] in R(0,1)
            #y_pred = T.max(p_y_given_x_sentence, axis=0) > 0.5  # note, max is just to coerce into proper shape
            #element_weights = T.outer(a, p_y_given_x_sentence)  # [ne, nc]

        elif pooling_method == 'attention2': # transform hidden nodes, sigmoid, then combine
            temp = T.dot(h, self.W_a) + self.b_a
            # softmax?
            a = T.exp(temp)/T.exp(temp).sum(axis=0, keepdims=True)  # [ne, minibatch_size, 1]: normalized over ne

            #a = T.nnet.softmax((T.dot(h, self.W_a) + self.b_a))
            a_sum = T.sum(a, axis=0)

            temp = T.nnet.sigmoid(T.dot(h, self.W_s) + self.b_s)  # [ne, minibatch_size, nc]
            p_y_given_x_sentence = T.sum(temp * T.repeat(a, nc, axis=2), axis=0)  # [minibatch_size, nc] in R(0,1)
            y_pred = p_y_given_x_sentence > 0.5
            #element_weights = T.repeat(a.T, nc, axis=1) * temp   # [ne, nc]
        elif pooling_method == 'mean':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, nc] in R(0,1)
            p_y_given_x_sentence = T.mean(s, axis=0)
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        elif pooling_method == 'max':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, minibatch_size, nc] in R(0,1)
            #s_shape = T.printing.Print('s_shape')(s.shape)
            #p_y_given_x_sentence = T.max(s_shape[0] * s, axis=0)
            p_y_given_x_sentence = T.max(s, axis=0)
            #p_y_given_x_sentence = T.printing.Print('p_y')(T.max(s, axis=0))
            #temp = T.printing.Print('p_y')(p_y_given_x_sentence)
            #y_pred = T.printing.Print('y_pred')(p_y_given_x_sentence > 0.5)
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        elif pooling_method == 'last':
            s = T.nnet.sigmoid((T.dot(h, self.W_s) + self.b_s))  # [n_elements, minibatch_size, nc] in R(0,1)
            p_y_given_x_sentence = s[-1, :, :]
            y_pred = p_y_given_x_sentence > 0.5
            element_weights = s
        else:
            sys.exit("Pooling method not recognized")



        # cost and gradients and learning rate
        lr = T.scalar('lr_main')
        lr_emb_fac = T.scalar('lr_emb')

        #sentence_nll = T.mean(T.sum(-T.log(y*p_y_given_x_sentence + (1-y)*(1-p_y_given_x_sentence)), axis=1))
        sentence_nll = T.sum(-T.log(y*p_y_given_x_sentence + (1-y)*(1-p_y_given_x_sentence)))

        sentence_gradients = T.grad(sentence_nll, self.params)

        clipped_grads = [T.clip(g, -1, 1) for g in sentence_gradients]
        grad_max = [T.max(g) for g in clipped_grads]
        #sentence_updates = OrderedDict((p, p - lr * T.max(g, 1)) for p, g in zip(self.params, [lr_emb_fac *
        #                                                                    sentence_gradients[0]]
        #                                                                    + sentence_gradients[1:]))
        sentence_updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, [lr_emb_fac *
                                                                            clipped_grads[0]]
                                                                            + clipped_grads[1:]))


        # theano functions to compile
        if extra_input_dims > 0:
            self.sentence_classify = theano.function(inputs=[idxs, mask, extra, minibatch_size], outputs=y_pred)
            self.sentence_train = theano.function(inputs=[idxs, mask, extra, y, lr, lr_emb_fac, minibatch_size],
                                                  outputs=[sentence_nll, a_sum],
                                                  updates=sentence_updates)
            self.get_gradients = theano.function(inputs=[idxs, mask, extra, y, minibatch_size],
                                                 outputs=grad_max)
            #if pooling_method == 'attention1' or pooling_method == 'attention2':
            #    self.a_sum_check = theano.function(inputs=[idxs, extra], outputs=a_sum)
            self.sentence_step_through = theano.function(inputs=[idxs, mask, extra, minibatch_size],
                                                         outputs=[h, self.W_s, self.b_s, p_y_given_x_sentence, s])

        else:
            self.sentence_classify = theano.function(inputs=[idxs, mask, minibatch_size], outputs=y_pred)
            self.sentence_train = theano.function(inputs=[idxs, mask, y, lr, lr_emb_fac, minibatch_size],
                                                  outputs=[sentence_nll, a_sum],
                                                  updates=sentence_updates)
            self.get_gradients = theano.function(inputs=[idxs, mask, y, minibatch_size],
                                                 outputs=grad_max)
            #if pooling_method == 'attention1' or pooling_method == 'attention2':
            #    self.a_sum_check = theano.function(inputs=[idxs, mask, minibatch_size], outputs=a_sum)
            self.sentence_step_through = theano.function(inputs=[idxs, mask, minibatch_size],
                                                         outputs=[h, self.W_s, self.b_s, p_y_given_x_sentence, s])


        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1))
                                         .dimshuffle(0, 'x')})

    def step_through(self, x, mask, window_size, extra_input_dims=0, extra=None):

        seq_len, minibatch_size, window_size = x.shape
        words = x
        mask = np.array(mask.T).astype('int32').reshape((seq_len, minibatch_size, 1))

        if extra_input_dims > 0:
            extra = np.array(extra).astype('int32').reshape((1, minibatch_size, extra_input_dims))
            return self.sentence_step_through(words, mask, extra, minibatch_size)
        else:
            return self.sentence_step_through(words, mask, minibatch_size)

    def classify(self, x, mask, window_size, extra_input_dims=0, extra=None):
        seq_len, minibatch_size, window_size = x.shape
        words = x
        mask = np.array(mask.T).astype('int32').reshape((seq_len, minibatch_size, 1))

        if extra_input_dims > 0:
            extra = np.array(extra).astype('int32').reshape((1, minibatch_size, extra_input_dims))
            return self.sentence_classify(words, mask, extra, minibatch_size)
        else:
            return self.sentence_classify(words, mask, minibatch_size)

    def train(self, x, mask, y, window_size, learning_rate, emb_lr_factor, extra_input_dims=0, extra=None):
        seq_len, minibatch_size, window_size = x.shape
        words = x
        mask = np.array(mask.T).astype('int32').reshape((seq_len, minibatch_size, 1))
        y = np.array(y).astype('int32')

        # train on these sentences and normalize
        if extra_input_dims > 0:
            extra = np.array(extra).astype('int32').reshape((1, minibatch_size, extra_input_dims))
            grads = self.get_gradients(words, mask, extra, y, minibatch_size)
            nll = self.sentence_train(words, mask, extra, y, learning_rate, emb_lr_factor, minibatch_size)
        else:
            grads = self.get_gradients(words, mask, y, minibatch_size)
            nll = self.sentence_train(words, mask, y, learning_rate, emb_lr_factor, minibatch_size)
        #print grads
        self.normalize()
        #for g in gradients:
        #    g = T.printing.Print('g')(T.sqrt(T.sum(g**2)))
        #    nll = nll*g/float(g)
        return nll

    def save(self, output_dir):
        for param in self.params:
            np.save(os.path.join(output_dir, param.name + '.npy'), param.get_value())

    def load(self, input_dir):
        for param in self.params:
            param.set_value(np.load(os.path.join(input_dir, param.name + '.npy')))

    def print_embeddings(self):
        for param in self.params:
            print param.name, param.get_value()



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




def main(params=None):

    if params is None:
        params = {
            'dataset': 'DRLD',
            'exp_name': 'best_minibatch_mod',
            'test_fold': 0,
            'n_dev_folds': 1,
            'min_doc_thresh': 1,
            'initialize_word_vectors': False,
            'vectors': 'anes_word2vec_300',  # default_word2vec_300, anes_word2vec_300, chars_word2vec_25, eye_1 ...
            'init_scale': 0.2,
            'add_OOV_dim': False,
            'win': 1,                   # size of context window
            'add_DRLD': False,
            'rnn_type': 'LSTM',        # basic, GRU, or LSTM
            'n_hidden': 50,             # size of hidden units
            'pooling_method': 'last',    # max, mean, or attention1/2
            'bidirectional': False,
            'bi_combine': 'concat',        # concat, max, or mean
            'train_embeddings': False,
            'lr': 0.025,                  # learning rate
            'lr_emb_fac': 0.2,            # factor to modify learning rate for embeddings
            'decay_delay': 5,           # number of epochs with no improvement before decreasing learning rate
            'decay_factor': 0.5,        # factor by which to multiply learning rate in case of delay
            'n_epochs': 100,
            'add_OOV_noise': False,
            'OOV_noise_prob': 0.01,
            'minibatch_size': 1,
            'classify_minibatch_size': 1,
            'ensemble': False,
            'save_model': True,
            'seed': 42,
            'verbose': 1,
            'reuse': False,
            'orig_T': 0.04,
            'tau': 0.01,
            'xavier_init': True
        }

    params = fh.read_json('/Users/dcard/Projects/CMU/ARK/guac/experiments/rnn/bayes_opt_rnn_LSTM_reuse_mod_34_rerun/params.txt')
    params['n_hidden'] = int(params['n_hidden'])

    keys = params.keys()
    keys.sort()
    for key in keys:
        print key, ':', params[key]

    # seed the random number generators
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    vector_type = params['vectors'].split('_')[0]
    params['word2vec_dim'] = int(params['vectors'].split('_')[-1])


    reuser = None
    if params['reuse']:
        reuser = reusable_holdout.ReuseableHoldout(T=params['orig_T'], tau=params['tau'])

    if params['dataset'] == 'DRLD':
        datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']
    elif params['dataset'] == 'MIP':
        datasets = ['MIP-Personal-1', 'MIP-Personal-2', 'MIP-Political-1', 'MIP-Political-2']
    elif params['dataset'] == 'MOLD':
        datasets = ['McCain-Likes', 'McCain-Dislikes', 'Obama-Likes', 'Obama-Dislikes']
    elif params['dataset'] == 'Primary':
        datasets = ['Obama-Primary', 'Clinton-Primary']
    elif params['dataset'] == 'General':
        datasets = ['Obama-General', 'McCain-General']
    else:
        datasets = [params['dataset']]

    np.random.seed(params['seed'])
    random.seed(params['seed'])

    best_valid_f1s = []
    best_true_valid_f1s = []
    best_test_f1s = []
    best_train_f1s = []

    test_prediction_arrays = []

    output_dir = fh.makedirs(defines.exp_dir, 'rnn', params['exp_name'])
    output_filename = fh.make_filename(output_dir, 'params', 'txt')
    fh.write_to_json(params, output_filename)

    for dev_fold in range(params['n_dev_folds']):
        print "dev fold =", dev_fold

        output_dir = fh.makedirs(defines.exp_dir, 'rnn', params['exp_name'], 'fold' + str(dev_fold))

        all_data, words2idx, items, all_labels = common.load_data(datasets, params['test_fold'], dev_fold,
                                                                  params['min_doc_thresh'])
        train_xy, valid_xy, test_xy = all_data
        train_lex, train_y = train_xy
        valid_lex, valid_y = valid_xy
        test_lex, test_y = test_xy

        train_lengths = [len(x) for x in train_lex]
        length_order = np.argsort(train_lengths)

        #if params['minibatch_size'] > 1 or params['classify_minibatch_size'] > 1:
        print "padding input with zeros"
        #all_data, all_masks = common.prepare_data(train_lex, valid_lex, test_lex, preset_max=100)
        all_data, all_masks = common.prepare_data(train_lex, valid_lex, test_lex)
        train_lex, valid_lex, test_lex = all_data
        train_masks, valid_masks, test_masks = all_masks
        #else:
        #    train_masks = [np.ones(len(x)).astype('int32') for x in train_lex]
        #    valid_masks = [np.ones(len(x)).astype('int32') for x in valid_lex]
        #    test_masks = [np.ones(len(x)).astype('int32') for x in test_lex]

        print "expanding x with context win dows"
        # Rejigger to convert x to contex win in advance
        train_x_win = expand_x_with_context_win(train_lex, params['win'])
        valid_x_win = expand_x_with_context_win(valid_lex, params['win'])
        test_x_win = expand_x_with_context_win(test_lex, params['win'])
        order = range(len(train_lex))
        print "done"

        train_items, dev_items, test_items = items
        vocsize = len(words2idx.keys())
        idx2words = dict((k, v) for v, k in words2idx.iteritems())
        best_test_predictions = None

        n_sentences = len(train_lex)
        print "vocsize = ", vocsize, 'n_train', n_sentences

        codes = all_labels.columns
        n_items, n_codes = all_labels.shape

        # get the words in the sentences for the test and validation sets
        words_valid = [map(lambda x: idx2words[x], w) for w in valid_lex]
        groundtruth_test = test_y[:]
        words_test = [map(lambda x: idx2words[x], w) for w in test_lex]

        #if vector_type == 'eye':
        #    initial_embeddings = np.eye(vocsize)
        #    emb_dim = initial_embeddings.shape[1]
        if params['initialize_word_vectors']:
            initial_embeddings = common.load_embeddings(params, words2idx)
            emb_dim = initial_embeddings.shape[1]
        else:
            initial_embeddings = None
            emb_dim = params['word2vec_dim']
        print "embedding dim =", emb_dim

        extra_input_dims = 0
        if params['add_DRLD']:
            #extra_input_dims = 4
            extra_input_dims = 2

        print "Building RNN"
        rnn = RNN(nh=params['n_hidden'],
                  nc=n_codes,
                  ne=vocsize,
                  de=emb_dim,
                  cs=params['win'],
                  extra_input_dims=extra_input_dims,
                  initial_embeddings=initial_embeddings,
                  init_scale=params['init_scale'],
                  rnn_type=params['rnn_type'],
                  train_embeddings=params['train_embeddings'],
                  pooling_method=params['pooling_method'],
                  bidirectional=params['bidirectional'],
                  bi_combine=params['bi_combine'],
                  xavier_init=params['xavier_init']
                  )

        # add extra dimensions to differentiate between paired datasets
        train_likes = [1 if re.search('Likes', i) else 0 for i in train_items]
        dev_likes = [1 if re.search('Likes', i) else 0 for i in dev_items]
        test_likes = [1 if re.search('Likes', i) else 0 for i in test_items]

        train_dem = [1 if re.search('Democrat', i) else 0 for i in train_items]
        dev_dem = [1 if re.search('Democrat', i) else 0 for i in dev_items]
        test_dem = [1 if re.search('Democrat', i) else 0 for i in test_items]

        """
        train_obama = [1 if re.search('Obama', i) else 0 for i in train_items]
        dev_obama = [1 if re.search('Obama', i) else 0 for i in dev_items]
        test_obama = [1 if re.search('Obama', i) else 0 for i in test_items]

        train_personal = [1 if re.search('Personal', i) else 0 for i in train_items]
        dev_personal = [1 if re.search('Personal', i) else 0 for i in dev_items]
        test_personal = [1 if re.search('Personal', i) else 0 for i in test_items]

        train_extra = [[train_likes[i], train_dem[i], train_obama[i], train_personal[i]] for i, t in enumerate(train_items)]
        dev_extra = [[dev_likes[i], dev_dem[i], dev_obama[i], dev_personal[i]] for i, t in enumerate(dev_items)]
        test_extra = [[test_likes[i], test_dem[i], test_obama[i], test_personal[i]] for i, t in enumerate(test_items)]
        """

        train_extra = [[train_likes[i], train_dem[i]] for i, t in enumerate(train_items)]
        dev_extra = [[dev_likes[i], dev_dem[i]] for i, t in enumerate(dev_items)]
        test_extra = [[test_likes[i], test_dem[i]] for i, t in enumerate(test_items)]


        ### LOAD
        rnn.load(output_dir)

        # train with early stopping on validation set
        best_f1 = -np.inf
        params['clr'] = params['lr']
        n_train = len(order)

        """
        for e in xrange(params['n_epochs']):
            # shuffle
            #shuffle([train_lex, train_y, train_extra, train_masks], params['seed'])   # shuffle the input data

            # sort by length on the first epoch
            if e == 0:
                order = length_order
                train_lex = [train_lex[j] for j in order]
                train_y = [train_y[j] for j in order]
                train_extra = [train_extra[j] for j in order]
                train_masks = [train_masks[j] for j in order]
            else:
                shuffle([order, train_lex, train_y, train_extra, train_masks], params['seed'])   # shuffle the input data
            params['ce'] = e                # store the current epoch
            tic = timeit.default_timer()

            ms = params['minibatch_size']
            n_train = len(train_lex)
            nll = 0

            #for i, orig_x in enumerate(train_lex):
            for iteration, i in enumerate(range(0, n_train, ms)):

                minibatch_x, minibatch_mask,\
                minibatch_extra, minibatch_y= select_minibatch(train_x_win, train_masks, train_extra, train_y,
                                                               params['win'], i, ms, order,
                                                               params['add_OOV_noise'], params['OOV_noise_prob'])

                n_elements, _, _ = minibatch_x.shape
                #if i > -1:
                #    print '\n'.join([' '.join([idx2words[idx] for idx in minibatch_x[:, k, 0].tolist()]) for
                #           k in range(ms)])
                #    print minibatch_y


                nll_i, a_sum = rnn.train(minibatch_x, minibatch_mask, minibatch_y, params['win'],
                                params['clr'] / float(n_elements) * 20,
                                params['lr_emb_fac'], extra_input_dims, minibatch_extra)

                nll += nll_i
                #rnn.train(x, mask, y, params['win'], params['clr'], params['lr_emb_fac'],
                #          extra_input_dims, extra)
                print '[learning] epoch %i >> %2.2f%%' % (
                    e, (i + 1) * 100. / float(n_sentences)),
                print 'completed in %.2f (sec), nll = %.2f, a_sum = %.1f <<\r' % (timeit.default_timer() - tic,
                                                                                  nll, np.max(a_sum)),
                sys.stdout.flush()

                if np.isnan(nll) or np.isinf(nll):
                    if best_f1 > 0:
                        break
                    else:
                        return {'loss': 1.0,
                                'final_test_f1': 0,
                                'valid_f1s': 0,
                                'true_valid_f1s': 0,
                                'train_f1s': 0,
                                'test_f1s': 0,
                                'status': STATUS_OK
                                }

            # evaluation // back into the real world : idx -> words
            print ""


            #print "true y", train_y[-1]
            #y_pred = rnn.classify(np.array(train_x_win[-1]).reshape((1, len(train_x_win[-1]))),
            #                      train_masks[-1], params['win'], extra_input_dims, train_extra[-1])[0]
            #print "pred y", y_pred

            #if params['pooling_method'] == 'attention1' or params['pooling_method'] == 'attention2':
            #    if extra_input_dims == 0:
            #        r = np.random.randint(0, len(train_lex))
            #        print r, rnn.a_sum_check(np.asarray(contextwin(train_lex[r], params['win'])).astype('int32'))

            """

        predictions_train = predict(n_train, params['classify_minibatch_size'], train_x_win, train_masks,
                                     train_y, params['win'], extra_input_dims, train_extra, rnn, order)
        n_valid = len(valid_lex)
        n_test = len(test_lex)
        predictions_valid = predict(n_valid, params['classify_minibatch_size'], valid_x_win, valid_masks,
                                    valid_y, params['win'], extra_input_dims, dev_extra, rnn)
        predictions_test = predict(n_test, params['classify_minibatch_size'], test_x_win, test_masks,
                                    test_y, params['win'], extra_input_dims, test_extra, rnn)

        """
        predictions_train = [rnn.classify(x, train_masks[i], params['win'],
                                          extra_input_dims, train_extra[i])[0] for i, x in enumerate(train_lex)]
        predictions_valid = [rnn.classify(x, valid_masks[i], params['win'],
                                          extra_input_dims, dev_extra[i])[0] for i, x in enumerate(valid_lex)]
        predictions_test = [rnn.classify(x, test_masks[i], params['win'],
                                         extra_input_dims, test_extra[i])[0] for i, x in enumerate(test_lex)]
        """

        train_f1 = common.calc_mean_f1(predictions_train, train_y)
        test_f1 = common.calc_mean_f1(predictions_test, test_y)
        valid_f1 = common.calc_mean_f1(predictions_valid, valid_y)

        output_dir = fh.makedirs(output_dir, 'responses')

        ms = 1


        for i in range(n_train):
            mb_x, mb_masks, mb_extra, mb_y = select_minibatch(train_x_win, train_masks, train_extra, train_y,
                                                              params['win'], i, ms, order=range(len(train_y)))

            h, W, b, p_y, s = rnn.step_through(mb_x, mb_masks, params['win'], extra_input_dims, mb_extra)

            temp = np.dot(h, W) + b
            s = 1.0/(1.0 + np.exp(-temp))
            output_filename = fh.make_filename(output_dir, train_items[i], 'csv')
            np.savetxt(output_filename, s[:, 0, :], delimiter=',')

        for i in range(n_valid):
            mb_x, mb_masks, mb_extra, mb_y = select_minibatch(valid_x_win, valid_masks, dev_extra, valid_y,
                                                              params['win'], i, ms, order=range(len(valid_y)))

            h, W, b, p_y, s = rnn.step_through(mb_x, mb_masks, params['win'], extra_input_dims, mb_extra)

            temp = np.dot(h, W) + b
            s = 1.0/(1.0 + np.exp(-temp))
            output_filename = fh.make_filename(output_dir, dev_items[i], 'csv')
            np.savetxt(output_filename, s[:, 0, :], delimiter=',')


        for i in range(n_test):
            mb_x, mb_masks, mb_extra, mb_y = select_minibatch(test_x_win, test_masks, test_extra, test_y,
                                                              params['win'], i, ms, order=range(len(test_y)))

            h, W, b, p_y, s = rnn.step_through(mb_x, mb_masks, params['win'], extra_input_dims, mb_extra)

            temp = np.dot(h, W) + b
            s = 1.0/(1.0 + np.exp(-temp))
            output_filename = fh.make_filename(output_dir, test_items[i], 'csv')
            np.savetxt(output_filename, s[:, 0, :], delimiter=',')



        print "train_f1 =", train_f1, "valid_f1 =", valid_f1, "test_f1 =", test_f1


def expand_x_with_context_win(lex, window_size):
    x = np.vstack(lex)
    n_items, seq_len = x.shape
    x_win = np.zeros([seq_len, n_items, window_size], dtype='int32')

    if window_size > 1:
        for i in range(n_items):
            x_win[:, i, :] = np.array(contextwin(list(x[i, :]), window_size), dtype='int32')
            #x_i =
        #x_win = [[np.array(w).astype('int32') for w in contextwin(list(x), window_size)] for x in lex]
    else:
        x_win[:, :, 0] = x.T

    print "x_win.shape", x_win.shape
    return x_win



def select_minibatch(x_win, masks, extra, y, window_size, i, minibatch_size, order=None, add_oov_noise=False, oov_noise_prob=0.0):
    n = len(masks)
    if order is None:
        order = range(n)
    ms = min(minibatch_size, n-i)
    if ms > 1:
        minibatch_mask = np.vstack([masks[j] for j in range(i, min(i+ms, n))])
        max_len = np.max(np.argmin(minibatch_mask, axis=1))
        if max_len == 0:
            max_len = len(masks[i])
        try:
            minibatch_mask = minibatch_mask[:, 0: max_len].reshape((ms, max_len))
        except:
            e = sys.exc_info()[0]
            print e
            print max_len
            print minibatch_mask
        minibatch_x = x_win[0: max_len, order[i: min(i+ms, n)], :]
        minibatch_extra = np.vstack([extra[j] for j in range(i, min(i+ms, n))])
        minibatch_y = np.vstack([y[j] for j in range(i, min(i+ms, n))])

    else:
        max_len = np.argmin(masks[i])
        if max_len == 0:
            max_len = len(masks[i])
        minibatch_mask = np.array(masks[i][0: max_len]).reshape((1, max_len))
        minibatch_x = x_win[0: max_len, order[i], :].reshape((max_len, 1, window_size))
        minibatch_extra = np.array(extra[i]).reshape((1, len(extra[i])))
        minibatch_y = np.array(y[i]).reshape((1, len(y[i])))

    if add_oov_noise:
        draws = np.random.rand(max_len, ms, window_size)
        minibatch_x = np.array(minibatch_x * np.array(draws > oov_noise_prob, dtype='int32'), dtype='int32')

    return minibatch_x, minibatch_mask, minibatch_extra, minibatch_y

def predict(n, ms, x_win, masks, y, window_size, extra_input_dims, extra, rnn, order=None):
    predictions = []
    for i in range(0, n, ms):

        mb_x, mb_masks, mb_extra, mb_y = select_minibatch(x_win, masks, extra, y, window_size, i, ms, order=order)

        if ms > 1:

            prediction = rnn.classify(mb_x, mb_masks, window_size, extra_input_dims, mb_extra)
            for p in prediction:
                predictions.append(p)
        else:

            prediction = rnn.classify(mb_x, mb_masks, window_size, extra_input_dims, mb_extra)
            predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    report = main()
    print report