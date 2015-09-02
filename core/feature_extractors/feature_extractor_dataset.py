
import re
import ast
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

import tokenizer
from feature_extractor_counts import FeatureExtractorCounts
from ..preprocessing import data_splitting as ds
from ..util import defines
from ..util import file_handling as fh


class FeatureExtractorDataset(FeatureExtractorCounts):

    def __init__(self, test_fold=0, dev_subfold=None):
        #print "Creating from arguments"
        name = 'dataset'
        prefix = '_d_'
        FeatureExtractorCounts.__init__(self, name, prefix, add_oov=True,
                                        min_doc_threshold=1,
                                        binarize=True,
                                        test_fold=test_fold,
                                        dev_subfold=dev_subfold)

    def get_dirname(self):
        return self.dirname

    def get_full_name(self):
        return fh.get_basename(self.dirname)

    def extract_features(self, write_to_file=True):
        print "Extracting ngram tokens:"

        if self.get_dev_subfold() is None:
            dev = 0
            train, dev, test = ds.get_all_splits(test_fold=self.get_test_fold(),
                                                 dev_subfold=self.get_dev_subfold())
            train = train + dev
        else:
            train, dev, test = ds.get_all_splits(test_fold=self.get_test_fold(),
                                                 dev_subfold=self.get_dev_subfold())

        all_items = train + dev + test

        label_files = fh.get_label_files()
        tokens = {}
        for f in label_files:
            print f
            Y = fh.read_csv(f)
            rids = Y.index
            for id in rids:
                tokens[id] = [self.get_prefix() + fh.get_basename(f)]

        # this is totally overkill for collecting the dataset names, but I think this is just easiest...
        vocab = self.make_vocabulary(tokens, all_items)

        feature_counts, oov_counts = self.extract_feature_counts(all_items, tokens, vocab)

        if write_to_file:
            vocab.write_to_file(self.get_vocab_filename())
            fh.write_to_json(all_items, self.get_index_filename(), sort_keys=False)
            fh.pickle_data(feature_counts, self.get_feature_filename())
            fh.write_to_json(oov_counts, self.get_oov_count_filename(), sort_keys=False)

        self.feature_counts = feature_counts
        self.index = all_items
        self.vocab = vocab
        self.oov_counts = oov_counts

    def get_counts(self):
        counts = self.feature_counts
        column_names = self.vocab.index2token
        index = pd.DataFrame(np.arange(len(self.index)), index=self.index, columns=['index'])
        return counts, index, column_names
