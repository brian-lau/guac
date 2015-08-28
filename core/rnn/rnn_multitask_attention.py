# adapted from https://github.com/lisa-lab/DeepLearningTutorials

from collections import OrderedDict
import copy
import os
import random
import timeit
import gensim

import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import common
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds
from ..util import defines
from ..util import file_handling as fh


# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(1500)


# utils functions
def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


# start-snippet-1
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
# end-snippet-1



# start-snippet-2

## CONVERT TO A SINGLE CLASS OUTPUT, since it gets softmaxed anyway?

class RNNSLU(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs, initial_embeddings=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        # parameters of the model
        if initial_embeddings is None:
            self.emb = theano.shared(name='embeddings',
                                     value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                      (ne+1, de))
                                     # add one for padding at the end
                                     .astype(theano.config.floatX))
        else:
            self.emb = theano.shared(name='embeddings',
                                     value=initial_embeddings.astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                 (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                 (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=np.zeros(nh,
                                                  dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=np.zeros(nc,
                                                 dtype=theano.config.floatX))
        #self.wa = theano.shared(name='wa',
        #                       value=0.2 * np.random.uniform(-1.0, 1.0,
        #                                                     (nh, nc))
        #                       .astype(theano.config.floatX))
        #self.ba = theano.shared(name='ba',
        #                       value=np.zeros(nc,
        #                                      dtype=theano.config.floatX))
        self.wa = theano.shared(name='wa',
                               value=0.2 * np.random.uniform(-1.0, 1.0,
                                                             (nh, 1))
                               .astype(theano.config.floatX))
        self.ba = theano.shared(name='ba',
                               value=0.0)
        self.h0 = theano.shared(name='h0',
                                value=np.zeros(nh,
                                                  dtype=theano.config.floatX))

        # bundle
        #self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]
        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.wa, self.ba, self.h0]
        #self.params = [self.wx, self.wh, self.w, self.bh, self.b, self.h0]
        # end-snippet-2
        # as many columns as context window size
        # as many lines as words in the sentence
        # start-snippet-3
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        #y = T.ivector('y_sentence')  # labels
        y = T.ivector('y')
        # end-snippet-3 start-snippet-4

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            #s_t = T.nnet.sigmoid(T.dot(h_t, self.w) + self.b)
            #return [h_t, s_t]
            return h_t

        """
        h, _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=self.h0,
                                n_steps=x.shape[0])
        """

        #[h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.h0, None], n_steps=x.shape[0])
        h, _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.h0], n_steps=x.shape[0])


        # Transform each hidden node into a [1,nc] vector and do a element-wise logit transform
        #s = T.nnet.sigmoid(T.dot(h, self.w) + self.b) # ne x nc
        #s = T.dot(h, self.w) + self.b # ne x nc

        option1 = False
        if option1:
            a = T.nnet.softmax((T.dot(h, self.wa) + self.ba).T) # 1 x ns: normalized (?)
            # SOFTMAX normalizes across the row (axis=1)

            a_sum = T.sum(a)

            p_y_given_x_sentence = T.nnet.sigmoid(T.dot(T.dot(a, h), self.w) + self.b) # 1 x nc

            # OLD option: take the max
            # and take the max over the transformed elements of the sequence to get a single [1,nc] vector
            #p_y_given_x_sentence = T.max(s, axis=0)

        else:
            a = T.nnet.softmax((T.dot(h, self.wa) + self.ba).T) # 1 x ns: normalized (?)
            # SOFTMAX normalizes across the row (axis=1)

            a_sum = T.sum(a)

            p_y_given_x_sentence = T.dot(a, T.nnet.sigmoid(T.dot(h, self.w) + self.b)) # 1 x nc

        # make a prediction for each class
        #y_pred = s > 0.5
        y_pred = p_y_given_x_sentence > 0.5

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = T.sum(-T.log(y*p_y_given_x_sentence + (1-y)*(1-p_y_given_x_sentence)))

        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))



        # end-snippet-5

        # theano functions to compile
        # start-snippet-6
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        # end-snippet-6 start-snippet-7
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                      self.emb /
                                                      T.sqrt((self.emb**2)
                                                             .sum(axis=1))
                                         .dimshuffle(0, 'x')})
        self.a_sum_check = theano.function(inputs=[idxs], outputs=a_sum)
        # end-snippet-7

        #self.print_params = theano.printing.Print('params:')(self.params)
        #self.print_params_fn = theano.function(inputs=[], outputs=self.print_params)


    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)
        labels = y

        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    def save(self, output_dir):
        for param in self.params:
            np.save(os.path.join(output_dir, param.name + '.npy'), param.get_value())

    def load(self, input_dir):
        for param in self.params:
            param.set_value(np.load(os.path.join(input_dir, param.name + '.npy')))

    def print_embeddings(self):
        #print self.emb.name, self.emb.get_value()
        for param in self.params:
            print param.name, param.get_value()


def main(params=None):

    if params is None:
        params = {
            'output_dir': None,
            'test_fold': 0,
            'dev_subfold': 0,
            'lr': 0.1,
            'verbose': 1,
            'decay': True,
            # decay on the learning rate if improvement stops
            'win': 3,
            # number of words in the context window
            'nhidden': 100,
            # number of hidden units
            'seed': 42,
            'word2vec_dim': 300,
            'bc_dim': 0,
            'extra_dims': 1,
            # dimension of word embedding
            'n_epochs': 40,
            # 60 is recommended
            'savemodel': True,
            'initialize_word_vectors': True,
            'custom_word2vec': True}
    print params

    # seed the random number generators
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    output_dir = params['output_dir']
    if output_dir is None:
        output_dir = fh.makedirs(defines.exp_dir, 'rnn', 'test')

    datasets = []

    train_items = []
    dev_items = []
    test_items = []
    label_list = []

    # load labels and train/test/split
    for d in datasets:
        train_items.extend(ds.get_train_documents(d, params['test_fold'], params['dev_subfold']))
        dev_items.extend(ds.get_dev_documents(d, params['test_fold'], params['dev_subfold']))
        test_items.extend(ds.get_test_documents(d, params['test_fold']))
        label_list.append(labels.get_datset_labels(d))

    all_labels = pd.concat(label_list, axis=0)

    # load word indices for all sentences
    all_lex = fh.read_json(fh.make_filename(defines.data_token_dir, 'ngrams_1_rnn_indices', 'json'))
    nItems, nCodes = all_labels.shape
    print nItems, nCodes
    # load a vocabulary index
    words2idx = fh.read_json(fh.make_filename(defines.data_token_dir, 'ngrams_1_rnn_vocab', 'json'))

    # build a train/test/validation set from the sentences
    train_lex = []
    train_y = []
    for item in train_items:
        train_lex.append(np.array(all_lex[item]).astype('int32'))
        train_y.append(np.array(all_labels.loc[item]).astype('int32'))
    valid_lex = []
    valid_y = []
    for item in dev_items:
        valid_lex.append(np.array(all_lex[item]).astype('int32'))
        valid_y.append(np.array(all_labels.loc[item]).astype('int32'))
    test_lex = []
    test_y = []
    for item in test_items:
        test_lex.append(np.array(all_lex[item]).astype('int32'))
        test_y.append(np.array(all_labels.loc[item]).astype('int32'))

    # invert the vocabulary index
    idx2word = dict((k, v) for v, k in words2idx.iteritems())

    vocsize = len(words2idx.keys())

    print "vocsize = ", vocsize
    n_sentences = len(train_lex)

    # get the words in the sentences for the test and validation sets
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    groundtruth_test = test_y[:]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    # load word vectors
    initial_embeddings = None
    if params['initialize_word_vectors']:
        if params['custom_word2vec']:
            # my word2vec vectors
            print "Loading custom word2vec vectors"
            vector_file = defines.my_word2vec_filename
            vectors = gensim.models.Word2Vec.load(vector_file)
        else:
            # standard word2vec
            print "Loading standard word2vec vectors"
            vector_file = defines.word2vec_vectors_filename
            vectors = gensim.models.Word2Vec.load_word2vec_format(vector_file, binary=True)

        print "Setting up initial embeddings"
        total_emb_dims = params['word2vec_dim'] + params['bc_dim'] + params['extra_dims']
        initial_embeddings = np.zeros([vocsize, total_emb_dims], dtype=float)
        #initial_embeddings = np.zeros([vocsize+1, params['word2vec_dim'] + params['bc_dim']], dtype=float)
        for w in words2idx.keys():
            i = words2idx[w]
            if w in vectors:
            #if w in vectors.index:
                #initial_embeddings[i, :params['word2vec_dim']] = vectors.loc[w]
                initial_embeddings[i, :params['word2vec_dim']] = vectors[w]
            # create a separate orthogonal dimension for OOV
            elif w == '__OOV__':
                initial_embeddings[i, -1] = 1
            else:
                print "no vector for", w
                initial_embeddings[i, :params['word2vec_dim']] = 0.05 * \
                    np.random.uniform(-1.0, 1.0, (1, params['word2vec_dim']))
            #if w in brown_index:
            #    initial_embeddings[i, params['word2vec_dim']:] = brown_vectors[brown_index[w]]

    print "Building RNN"
    rnn = RNNSLU(nh=params['nhidden'],
                 nc=nCodes,
                 ne=vocsize,
                 de=total_emb_dims,
                 cs=params['win'],
                 initial_embeddings=initial_embeddings)

    # train with early stopping on validation set
    best_f1 = -np.inf
    params['clr'] = params['lr']
    for e in xrange(params['nepochs']):

        # shuffle
        shuffle([train_lex, train_y], params['seed'])

        params['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            rnn.train(x, y, params['win'], params['clr'])
            print '[learning] epoch %i >> %2.2f%%' % (
                e, (i + 1) * 100. / nsentences),
            print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic),
            sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print ""


        print rnn.classify(np.asarray(contextwin(train_lex[0], params['win'])).astype('int32'))
        print rnn.a_sum_check(np.asarray(contextwin(train_lex[0], params['win'])).astype('int32'))


        predictions_train = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                             for x in train_lex]
        predictions_test = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                            for x in test_lex]
        predictions_valid = [np.max(rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32')), axis=0)
                             for x in valid_lex]


        """
        predictions_train = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32'))
                             for x in train_lex]
        predictions_test = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32'))
                            for x in test_lex]
        predictions_valid = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32'))
                             for x in valid_lex]
        """

        #detailed_valid_test = [rnn.classify(np.asarray(contextwin(x, params['win'])).astype('int32'))
        #                      for x in valid_lex]

        train_f1 = common.calc_mean_f1(predictions_train, train_y)
        test_f1 = common.calc_mean_f1(predictions_test, test_y)
        valid_f1 = common.calc_mean_f1(predictions_valid, valid_y)

        question_f1s = []
        question_pps = []

        print "train_f1 =", train_f1, "valid_f1 =", valid_f1, "test_f1 =", test_f1

        if valid_f1 > best_f1:
            if params['savemodel']:
                rnn.save(output_dir)
                #save_predictions(detailed_valid_test, groundtruth_valid, words_valid, output_dir + '/current.valid.txt')
                common.write_predictions(datasets, predictions_valid, dev_items, output_dir)

            best_rnn = copy.deepcopy(rnn)
            best_f1 = valid_f1

            if params['verbose']:
                print('NEW BEST: epoch', e,
                      'valid f1', valid_f1,
                      'best test f1', test_f1)

            params['tf1'] = test_f1
            params['vf1'] = valid_f1
            #params['vf1'], params['tf1'] = valid_f1, test_f1
            #params['vp'], params['tp'] = valid_prec, test_prec
            #params['vr'], params['tr'] = valid_recall, test_recall
            params['be'] = e

            #subprocess.call(['mv', output_dir + '/current.test.txt',
            #                 output_dir + '/best.test.txt'])
            #subprocess.call(['mv', output_dir + '/current.valid.txt',
            #                 output_dir + '/best.valid.txt'])

        # learning rate decay if no improvement in 10 epochs
        if params['decay'] and abs(params['be']-params['ce']) >= 10:
            params['clr'] *= 0.5
            rnn = best_rnn

        if params['clr'] < 1e-5:
            break

        if best_f1 == 1.0:
            break

    best_rnn.print_embeddings()

    print('BEST RESULT: epoch', params['be'],
          'valid F1', params['vf1'],
          'best test F1', params['tf1'],
          'with the model', output_dir)



if __name__ == '__main__':
    main()