import ast
from collections import Counter
from scipy import sparse
import random

import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor
import vocabulary_with_counts
from ..util import file_handling as fh

class FeatureExtractorCounts(FeatureExtractor):

    def __init__(self, name, prefix, add_oov=True, min_doc_threshold=1,
                 binarize=True, test_fold=0, dev_subfold=None):
        FeatureExtractor.__init__(self, name, prefix, 'counts',
                                  test_fold, dev_subfold)
        self.params['add_oov'] = ast.literal_eval(str(add_oov))
        self.params['min_doc_threshold'] = int(min_doc_threshold)
        self.params['binarize'] = ast.literal_eval(str(binarize))
        self.feature_counts = None
        self.index = None
        self.vocab = None
        self.oov_counts = None
        FeatureExtractorCounts.extend_dirname(self)

    def get_add_oov(self):
        return self.params['add_oov']

    def get_min_doc_threshold(self):
        return self.params['min_doc_threshold']

    def get_binarize(self):
        return self.params['binarize']

    def extend_dirname(self):
        self.dirname = self.dirname + ',' + str(self.params['add_oov']) + ',' \
            + str(self.params['min_doc_threshold']) + ',' + str(self.params['binarize'])

    @classmethod
    def parse_dirname(cls, dirname):
        name, feature_type, test_fold, dev_subfold, parts = FeatureExtractor.parse_dirname(dirname)
        assert feature_type == 'counts'
        assert len(parts) > 2
        add_oov = ast.literal_eval(parts[0])
        min_doc_threshold = int(parts[1])
        binarize = ast.literal_eval(parts[2])
        if len(parts) > 2:
            extra = parts[3:]
        else:
            extra = None
        return name, feature_type, test_fold, dev_subfold, add_oov, min_doc_threshold, binarize, extra

    def get_feature_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'counts', 'pkl')

    def get_oov_count_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'oov_counts', 'json')

    def make_vocabulary(self, tokens, items, verbose=True):
        if verbose:
            print "Making vocabulary for", self.get_name()

        assert self.get_type() == 'counts'

        add_oov = self.get_add_oov()

        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=add_oov)

        for item in items:
            vocab.add_tokens(tokens[item])

        if verbose:
            print "Vocabulary size before pruning:", len(vocab)

        vocab.prune(min_doc_threshold=self.get_min_doc_threshold())

        if verbose:
            print "Vocabulary size after pruning:", len(vocab)

        return vocab

    def extract_feature_counts(self, items, tokens, vocab):
        n_items = len(items)
        n_features = len(vocab)

        #feature_counts = pd.DataFrame(np.zeros([n_items, n_features]),
        #                              index=items, columns=vocab.get_all_tokens(),
        #                              dtype=dtype)

        row_starts_and_ends = [0]
        column_indices = []
        values = []
        oov_counts = []

        for item in items:
            #token_counts = Counter(tokens[item])
            #token_keys = token_counts.keys()
            #token_indices = vocab.get_indices(token_keys)

            # get the index for each token
            token_indices = vocab.get_indices(tokens[item])

            # count how many times each index appears
            token_counter = Counter(token_indices)
            token_keys = token_counter.keys()
            token_counts = token_counter.values()

            # put it into the from of a sparse matix
            column_indices.extend(token_keys)
            if self.get_binarize():
                values.extend([1]*len(token_counts))
            else:
                values.extend(token_counts)

            oov_counts.append(token_counter.get(vocab.oov_index, 0))
            row_starts_and_ends.append(len(column_indices))

        #if self.get_binarize():
        #    dtype = 'int8'
        #else:
        #    dtype = 'int32'
        dtype = 'int32'

        feature_counts = sparse.csr_matrix((values, column_indices, row_starts_and_ends), dtype=dtype)

        assert feature_counts.shape[0] == n_items
        assert feature_counts.shape[1] == n_features

        return feature_counts, oov_counts


    def load_from_files(self, debug=False, debug_index=None):
        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=self.get_add_oov(),
                                                       read_from_filename=self.get_vocab_filename())
        index = fh.read_json(self.get_index_filename())
        feature_counts = fh.unpickle_data(self.get_feature_filename())
        oov_counts = fh.read_json(self.get_oov_count_filename())

        # TESTING
        if debug:
            if debug_index is None:
                item_index = random.randint(0, len(index))
            else:
                item_index = debug_index
            item = index[item_index]
            counts = feature_counts[item_index, :]

            print item
            print counts.indices
            print counts.data
            print vocab.get_tokens(counts.indices)
            print oov_counts[item_index]

        self.feature_counts = feature_counts
        self.index = index
        self.vocab = vocab
        self.oov_counts = oov_counts

    def get_counts(self):
        counts = self.feature_counts
        column_names = self.vocab.index2token
        index = pd.DataFrame(np.arange(len(self.index)), index=self.index, columns=['index'])
        return counts, index, column_names


