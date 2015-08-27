import re
import ast
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

import tokenizer
from feature_extractor_counts import FeatureExtractorCounts
from .. preprocessing import data_splitting as ds
from .. util import defines
from .. util import file_handling as fh


class FeatureExtractorCountsNgrams(FeatureExtractorCounts):

    def __init__(self, test_fold=0, dev_subfold=None, n=1, min_doc_threshold=1, binarize=True,
                 concat_oov_counts=False, append_dataset=False):
        #print "Creating from arguments"
        name = 'ngrams'
        prefix = '_n' + str(n) + '_'
        FeatureExtractorCounts.__init__(self, name, prefix, add_oov=True,
                                        min_doc_threshold=min_doc_threshold,
                                        binarize=binarize,
                                        test_fold=test_fold,
                                        dev_subfold=dev_subfold)
        self.params['n'] = int(n)
        self.params['concat_oov_counts'] = ast.literal_eval(str(concat_oov_counts))
        self.params['append_dataset'] = ast.literal_eval(str(append_dataset))
        FeatureExtractorCountsNgrams.extend_dirname(self)

    @classmethod
    def from_files(cls, dirname):
        #print "Loading from files", dirname
        name, feature_type, test_fold, dev_subfold,\
            add_oov, min_doc_threshold, binarize, extra = FeatureExtractorCounts.parse_dirname(dirname)
        assert len(extra) == 1
        n = int(extra[0])
        assert name == 'ngrams'
        return cls(test_fold=test_fold, dev_subfold=dev_subfold, n=n, min_doc_threshold=min_doc_threshold,
                   binarize=binarize)

    def get_n(self):
        return self.params['n']

    def extend_dirname(self):
        self.dirname = self.dirname + '_' + str(self.get_n()) \
            + '_' + str(self.params['concat_oov_counts']) \
            + '_' + str(self.params['append_dataset'])

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

        responses = fh.read_json(defines.data_raw_text_file)

        label_files = fh.get_label_files()
        tokens = {}
        for f in label_files:
            print f
            self.extract_tokens_from_file(responses, f, self.get_n(), tokens)

        # TRYING THIS: JUST USE ALL THE DATA TO BUILD THE VOCAB...
        #vocab = self.make_vocabulary(tokens, train)
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


    def extract_tokens_from_file(self, responses, input_filename, n, token_dict):
        Y = fh.read_csv(input_filename)
        rids = Y.index
        dataset = fh.get_basename(input_filename)

        for rid in rids:
            text = responses[rid].lower()
            text = text.lstrip()
            text = text.rstrip()
            text = text.lstrip('/')
            text = re.sub('<', '', text)
            text = re.sub('>', '', text)
            text = re.sub('-', ' - ', text)
            text = re.sub('_', ' - ', text)
            tokens = []
            paragraphs = re.split('[/\\\\]', text)
            paragraphs = [p for p in paragraphs if p != '']

            count = 0
            for p in paragraphs:
                count += 1
                sentences = tokenizer.split_sentences(p)
                for s in sentences:
                    sent_tokens = tokenizer.make_ngrams(s, n)
                    sent_tokens = [t.rstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                    sent_tokens = [t.lstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                    tokens = tokens + sent_tokens

            tokens = [self.get_prefix() + t for t in tokens]
            if self.params['append_dataset']:
                tokens = [t + '_' + dataset for t in tokens]
            token_dict[rid] = tokens

    def get_counts(self):
        if self.params['concat_oov_counts']:
            print self.feature_counts.shape
            print sparse.csr_matrix(self.oov_counts).T.shape
            counts = sparse.csr_matrix(sparse.hstack([self.feature_counts,
                                                      sparse.csr_matrix(self.oov_counts).T]))
            column_names = self.vocab.index2token + ['__OOV__counts__']
        else:
            counts = self.feature_counts
            column_names = self.vocab.index2token
        index = pd.DataFrame(np.arange(len(self.index)), index=self.index, columns=['index'])
        return counts, index, column_names


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='ngrams', default=1,
                      help='n for ngrams; default=%default')
    parser.add_option('-m', dest='min_doc_thresh', default=1,
                      help='Minimum document threshold; default=%default')
    parser.add_option('-b', dest='binarize', action="store_true", default=False,
                      help='Binarize counts; default=%default')
    parser.add_option('-d', dest='dev_subfold', default=0,
                      help='dev subfold to not learn from; default=%default')
    parser.add_option('-t', dest='test_fold', default=0,
                      help='test fold to not learn from; default=%default')

    (options, args) = parser.parse_args()
    n = int(options.ngrams)
    min_doc_thresh = int(options.min_doc_thresh)
    binarize = options.binarize
    dev_subfold = int(options.dev_subfold)
    test_fold = int(options.test_fold)

    extractor = FeatureExtractorCountsNgrams(test_fold=test_fold, dev_subfold=dev_subfold, n=n,
                                             min_doc_threshold=min_doc_thresh, binarize=binarize)
    extractor.extract_features(write_to_file=True)


if __name__ == '__main__':
    main()
