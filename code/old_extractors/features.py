from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd

from feature_extractors import vocabulary
from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog <feature_definition>"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dataset', default='',
                      help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    parser.add_option('-o', dest='output_dir', default='',
                      help='Output directory (experiment specific); (not written to file by default)')
    parser.add_option('-t', dest='test_fold', default=0,
                      help='Test fold; default=%default')
    parser.add_option('-v', dest='dev_subfold', default=0,
                      help='Dev subfold; default=%default')

    (options, args) = parser.parse_args()

    feature_definition_name = args[0]
    test_fold = options.test_fold
    dev_subfold = options.dev_subfold
    output_dir = options.output_dir


def make_feature_definition(feature_name,
                            prefix,
                            filename=None,
                            min_doc_threshold=1,
                            binarize=True,
                            feature_type='tokens'):

    feature_definition = {'name': feature_name, 'prefix': prefix, 'type': feature_type}
    if min_doc_threshold is not None:
        feature_definition['min_threshold'] = min_doc_threshold
    if binarize is not None:
        feature_definition['binarize'] = binarize

    output_dir = fh.makedirs(defines.data_featuredefns_dir)
    if filename is None: filename = feature_name
    basename = fh.get_basename(filename)
    output_filename = fh.make_filename(output_dir, basename, 'json')
    fh.write_to_json(feature_definition, output_filename)


#def read_feature_definition(filename):
#    feature_filename = fh.make_filename(defines.data_featuredefns_dir, filename, 'json')
#    feature_definition = fh.read_json(feature_filename)
#    return feature_definition

def load_feature_definition(feature_definition_name):
    # load the feature definition
    feature_definition_filename = fh.make_filename(defines.data_featuredefns_dir, feature_definition_name, 'json')
    #feature_definition = read_feature_definition(feature_definition_filename)
    feature_definition = fh.read_json(feature_definition_filename)
    return feature_definition

def load_tokens(feature_definition):
    # load the tokens for this feature
    input_filename = get_tokens_filename(feature_definition)
    tokens = fh.read_json(input_filename)
    return tokens

def load_values(feature_definition):
    # load the values for this feature
    input_filename = get_values_filename(feature_definition)
    tokens = fh.read_csv(input_filename)
    return tokens

def get_tokens_filename(feature_definition):
    feature_name = feature_definition['name']
    tokens_dir = fh.makedirs(defines.data_token_dir)
    return fh.make_filename(tokens_dir, feature_name, 'json')

def get_values_filename(feature_definition):
    feature_name = feature_definition['name']
    values_dir = fh.makedirs(defines.data_values_dir)
    return fh.make_filename(values_dir, feature_name, 'csv')


def make_vocabulary(feature_definition_name, items, verbose=True):

    feature_definition = load_feature_definition(feature_definition_name)
    feature_type = feature_definition['type']
    prefix = feature_definition['prefix']

    if verbose:
        print "Making vocabulary for", feature_definition_name

    if feature_type == 'values':
        values = load_values(feature_definition)
        vocab = vocabulary.Vocab(prefix, tokens_to_add=values.columns, add_oov=False)
        print "Vector size:", len(vocab)

    elif feature_type == 'tokens':
        if 'add_oov' in feature_definition:
            add_oov = feature_definition['add_oov']
        else:
            add_oov = True

        tokens = load_tokens(feature_definition)
        # count the number of occurrences of each token in the training set (specified by items)
        token_counts = Counter()
        token_doc_counts = Counter()

        for item in items:
            token_counts.update(tokens[item])
            token_doc_counts.update(set(tokens[item]))

        if verbose:
            print "Vocabulary size before pruning:", len(token_counts)

        min_threshold = feature_definition['min_threshold']
        valid_tokens = [t for (t, c) in token_counts.items() if c >= min_threshold]

        vocab = vocabulary.Vocab(prefix, tokens_to_add=valid_tokens, add_oov=add_oov)
        if verbose:
            print "Vocabulary size after pruning:", len(vocab)
    else:
        vocab = None

    return vocab

def extract_feature(feature_definition_name, items, vocab, verbose=True):
    feature_definition = load_feature_definition(feature_definition_name)
    feature_type = feature_definition['type']

    if feature_type == 'tokens':
        values = extract_feature_counts(feature_definition_name, items, vocab, verbose)
    else:
        values = extract_feature_values(feature_definition_name, items, verbose)

    return values

def extract_feature_counts(feature_definition_name, items, vocab, verbose=True):
    feature_definition = load_feature_definition(feature_definition_name)
    if verbose:
        print "Extracting features for", feature_definition_name
    all_tokens = load_tokens(feature_definition)

    n_items = len(items)
    n_features = len(vocab)

    if feature_definition['binarize']:
        feature_counts = pd.DataFrame(np.zeros([n_items, n_features]),
                                      index=items, columns=vocab.get_all_tokens(),
                                      dtype='bool')
    else:
        feature_counts = pd.DataFrame(np.zeros([n_items, n_features]),
                                      index=items, columns=vocab.get_all_tokens(),
                                      dtype='int32')

    for item in items:
        token_counts = Counter(all_tokens[item])
        tokens = token_counts.keys()
        indices = vocab.get_indices(tokens)
        if feature_definition['binarize']:
            feature_counts.loc[item][indices] = 1
        else:
            ## NOT SURE IF THIS IS WORKING OR NOT...
            feature_counts.loc[item][indices] += 1
            counts = np.array(token_counts.values()) - 1
            while np.sum(counts) > 0:
                token_counts = {t: c-1 for (t, c) in token_counts.items() if c-1 > 0}
                tokens = token_counts.keys()
                counts = token_counts.values()
                indices = vocab.get_indices(tokens)
                feature_counts.loc[item][indices] += 1
                counts = np.array(token_counts.values()) - 1

    return feature_counts


def extract_feature_values(feature_definition_name, items, verbose=True):
    feature_definition = load_feature_definition(feature_definition_name)
    if verbose:
        print "Extracting features for", feature_definition_name
    all_values = load_values(feature_definition)
    feature_values = all_values.loc[items]
    return feature_values


if __name__ == '__main__':
    main()
