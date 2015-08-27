import os
import sys
import gzip
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

from old_extractors import features
from old_extractors.features import tokenizer
from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog <vector-file.txt.gz>"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='prefix', default='_ndv_',
                      help='Token prefix; default=%default')
    (options, args) = parser.parse_args()

    prefix = options.prefix
    input_dir = defines.data_raw_labels_dir
    files = fh.ls(input_dir, '*.csv')

    sys.exit("THIS IS NOT WORKING RIGHT NOW")

    if len(args) < 1:
        sys.exit('Please specify a set of vectors')
    vector_filename = args[0]

    # load the vectors from a file and determine their size
    print "Loading vectors"
    vector_filename = os.path.join(defines.non_distributional_vectors_dir, vector_filename)
    if not os.path.exists(vector_filename):
        print vector_filename
        sys.exit("Cannot find vector file")

    words = get_word_subset(files)

    vectors = {}
    vector_size = 0
    with gzip.open(vector_filename, 'r') as input_file:
        count = 0
        for line in input_file:
            parts = line.split()
            word = parts[0]
            vector_size = len(parts[1:])
            count += 1
            if word in words:
                print word, count
                vector = sparse.csr_matrix(parts[1:], dtype=np.int8)
                vectors[word] = vector

    print "Extracting vector values:"
    data_matrices = []
    for f in files:
        print f
        data_matrices.append(extract_vector_values(f, vectors, vector_size, prefix))

    data = pd.concat(data_matrices, axis=0)

    non_zero_cols = data.sum(axis=0) > 0

    print data.shape
    data = data.loc[:, non_zero_cols]
    print data.shape

    output_dir = fh.makedirs(defines.data_values_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(vector_filename), 'csv')
    data.to_csv(output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(vector_filename), prefix,
                                     filename=get_feature_name(vector_filename)+'_default',
                                     binarize=False, min_doc_threshold=0,
                                     feature_type='values')


def get_feature_name(vector_filename):
    return 'non-distributional' + fh.get_basename(vector_filename)


def get_word_subset(f):
    print "Building subset of words"
    words = set()
    responses = fh.read_json(defines.data_raw_text_file)

    Y = fh.read_csv(f)
    rids = Y.index

    for rid in rids:
        text = responses[rid].lower()
        tokens = []
        paragraphs = text.split('/')
        paragraphs = [p for p in paragraphs if p != '']
        for p in paragraphs:
            sentences = tokenizer.split_sentences(p)
            for s in sentences:
                tokens = tokens + tokenizer.make_ngrams(s, n=1)
                words.update(set(tokens))

    print len(words), "words"
    return words



def extract_vector_values(input_filename, vectors, vector_size, prefix):

    Y = fh.read_csv(input_filename)
    rids = Y.index
    n_items = len(rids)

    responses = fh.read_json(defines.data_raw_text_file)

    keys = vectors.keys()

    col_names = [prefix + str(v) for v in np.arange(vector_size)]
    data = pd.DataFrame(np.zeros([n_items, vector_size]), index=rids, columns=col_names)

    for rid in rids:
        text = responses[rid].lower()
        tokens = []
        paragraphs = text.split('/')
        paragraphs = [p for p in paragraphs if p != '']
        for p in paragraphs:
            sentences = tokenizer.split_sentences(p)
            for s in sentences:
                tokens = tokens + tokenizer.make_ngrams(s, n=1)

        item_vector = sparse.csr_matrix(np.zeros([1, vector_size], dtype=np.int8))
        for t in tokens:
            if t in vectors:
                vector = vectors[t]
                item_vector += vector

        data.loc[rid] = item_vector.todense()

    return data


if __name__ == '__main__':
    main()
