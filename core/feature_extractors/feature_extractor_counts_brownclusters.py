import re

import tokenizer
from feature_extractor_counts import FeatureExtractorCounts
from ..preprocessing import data_splitting as ds
from ..util import defines
from ..util import file_handling as fh


class FeatureExtractorCountsBrownClusters(FeatureExtractorCounts):

    def __init__(self, test_fold=0, dev_subfold=None, binarize=False, clusters=''):
        #print "Creating from arguments"
        name = 'brownclusters'
        prefix = '_bc-' + clusters + '_'
        FeatureExtractorCounts.__init__(self, name, prefix, add_oov=True,
                                        min_doc_threshold=1,
                                        binarize=binarize,
                                        test_fold=test_fold,
                                        dev_subfold=dev_subfold)
        self.params['clusters'] = clusters
        FeatureExtractorCountsBrownClusters.extend_dirname(self)

    def extend_dirname(self):
        self.dirname = self.dirname + ',' + self.params['clusters']

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

        responses = fh.read_json(defines.data_normalized_text_file)

        cluster_filename = fh.make_filename(defines.resources_clusters_dir, self.params['clusters'], 'json')
        cluster_dict = fh.read_json(cluster_filename)['index']

        label_files = fh.get_label_files()
        tokens = {}
        for f in label_files:
            print f
            self.extract_tokens_from_file(responses, f, 1, cluster_dict, tokens)

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

    def extract_tokens_from_file(self, responses, input_filename, n, cluster_dict, token_dict):
        Y = fh.read_csv(input_filename)
        rids = Y.index

        for rid in rids:
            text = responses[rid].lower()
            text = text.lstrip()
            text = text.rstrip()
            tokens = []

            sentences = tokenizer.split_sentences(text)
            for s in sentences:
                sent_tokens = tokenizer.make_ngrams(s, n)
                sent_tokens = [t.rstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                sent_tokens = [t.lstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                sent_tokens = sent_tokens + ['__ENDS__']
                tokens = tokens + sent_tokens

            tokens = [self.get_prefix() + cluster_dict[t] for t in tokens if t in cluster_dict]
            token_dict[rid] = tokens


def main():
    pass

if __name__ == '__main__':
    main()
