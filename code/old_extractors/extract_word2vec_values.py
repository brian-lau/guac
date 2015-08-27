import sys
from optparse import OptionParser

import gensim
import numpy as np
import pandas as pd

from old_extractors import features
from old_extractors.features import tokenizer
from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog [add|mean|max]"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    input_dir = defines.data_raw_labels_dir
    files = fh.ls(input_dir, '*.csv')

    if len(args) < 1:
        sys.exit('Please specify add, mean, or max')
    op = args[0]
    if not (op == 'add' or op == 'mean' or op == 'max'):
        sys.exit('Please specify add, mean, or max')

    # load the vectors from a file and determine their size
    print "Loading vectors"
    vector_file = defines.word2vec_vectors_filename
    vectors = gensim.models.Word2Vec.load_word2vec_format(vector_file, binary=True)

    print "Extracting word2vec values:"
    data_matrices = []
    for f in files:
        print f
        data_matrices.append(extract_word2vec_values(f, op, vectors))

    data = pd.concat(data_matrices, axis=0)

    output_dir = fh.makedirs(defines.data_values_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(op), 'csv')
    data.to_csv(output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(op), get_prefix(op),
                                     filename=get_feature_name(op)+'_default',
                                     binarize=False, min_doc_threshold=0,
                                     feature_type='values')


def get_feature_name(op):
    return 'word2vec' + '_' + op


def get_prefix(op):
    return '_w2v' + op + '_'


def extract_word2vec_values(input_filename, op, vectors):

    Y = fh.read_csv(input_filename)
    rids = Y.index
    n_items = len(rids)

    responses = fh.read_json(defines.data_raw_text_file)

    prefix = get_prefix(op)
    vector_size = len(vectors['the'])

    if op == 'max':
        col_names = [prefix + str(v) + 'pos' for v in np.arange(vector_size)] + \
                        [prefix + str(v) + 'neg' for v in np.arange(vector_size)]
        data = pd.DataFrame(np.zeros([n_items, vector_size*2]), index=rids, columns=col_names)
    else:
        col_names = [prefix + str(v) for v in np.arange(vector_size)]
        data = pd.DataFrame(np.zeros([n_items, vector_size]), index=rids, columns=col_names)

    for rid in rids:
        if op == 'max':
            item_vector = np.zeros(vector_size*2)
        else:
            item_vector = np.zeros(vector_size)
        text = responses[rid].lower()
        tokens = []
        paragraphs = text.split('/')
        paragraphs = [p for p in paragraphs if p != '']
        for p in paragraphs:
            sentences = tokenizer.split_sentences(p)
            for s in sentences:
                tokens = tokens + tokenizer.make_ngrams(s, n=1)
        count = 0
        for t in tokens:
            if t in vectors:
                vector = vectors[t]
                if op == 'add' or op == 'mean':
                    item_vector += vector
                    count += 1
                elif op == 'max':
                    item_vector[:vector_size] = np.maximum(item_vector[:vector_size], vector)
                    item_vector[vector_size:] = np.minimum(item_vector[vector_size:], vector)
        if op == 'mean' and count > 0:
                item_vector /= float(count)

        data.loc[rid] = item_vector

    return data


if __name__ == '__main__':
    main()
