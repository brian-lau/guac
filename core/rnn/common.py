import os
import codecs
import gensim

import numpy as np
import pandas as pd

import extract_ngram_tokens_for_rnn
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds
from ..experiment import evaluation
from ..util import defines
from ..util import file_handling as fh


def load_data(datasets, test_fold, dev_subfold, min_doc_thresh, test_item=None):
    train_items = []
    dev_items = []
    test_items = []
    label_list = []

    # load labels and train/test/split
    for d in datasets:
        train_items.extend(ds.get_train_documents(d, test_fold, dev_subfold))
        dev_items.extend(ds.get_dev_documents(d, test_fold, dev_subfold))
        test_items.extend(ds.get_test_documents(d, test_fold))
        label_list.append(labels.get_dataset_labels(d))

    if test_item is not None:
        test_items = [test_item]

    items = (train_items, dev_items, test_items)

    all_labels = pd.concat(label_list, axis=0)

    basename = extract_ngram_tokens_for_rnn.get_feature_name(n=1, m=min_doc_thresh)
    all_lex_filename = fh.make_filename(defines.data_rnn_dir, basename, 'json')
    if not os.path.exists(all_lex_filename):
        extract_ngram_tokens_for_rnn.prepare_data_for_rnn(datasets, n=1, min_threshold=min_doc_thresh)

    # load word indices for all sentences
    all_lex = fh.read_json(fh.make_filename(defines.data_rnn_dir, basename + '_indices', 'json'))
    # load a vocabulary index
    words2idx = fh.read_json(fh.make_filename(defines.data_rnn_dir, basename + '_vocab', 'json'))

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

    data = ((train_lex, train_y), (valid_lex, valid_y), (test_lex, test_y))

    return data, words2idx, items, all_labels


def load_char_data(datasets, test_fold, dev_subfold):
    train_items = []
    dev_items = []
    test_items = []
    label_list = []

    # load labels and train/test/split
    for d in datasets:
        train_items.extend(ds.get_train_documents(d, test_fold, dev_subfold))
        dev_items.extend(ds.get_dev_documents(d, test_fold, dev_subfold))
        test_items.extend(ds.get_test_documents(d, test_fold))
        label_list.append(labels.get_dataset_labels(d))

    items = (train_items, dev_items, test_items)

    all_labels = pd.concat(label_list, axis=0)

    basename = 'chars_rnn'
    all_lex_filename = fh.make_filename(defines.data_rnn_dir, basename, 'json')

    # load word indices for all sentences
    all_lex = fh.read_json(fh.make_filename(defines.data_rnn_dir, basename + '_indices', 'json'))
    # load a vocabulary index
    words2idx = fh.read_json(fh.make_filename(defines.data_rnn_dir, basename + '_vocab', 'json'))

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

    data = ((train_lex, train_y), (valid_lex, valid_y), (test_lex, test_y))

    return data, words2idx, items, all_labels



# this creates masks such that all input sequences are the same length
def prepare_data(train_x, valid_x, test_x, preset_max=None):

    all_x = train_x + valid_x + test_x
    lengths = [len(x) for x in all_x]
    max_len = np.max(lengths)
    if preset_max is not None:
        max_len = min(max_len, preset_max)

    train_x, train_mask = prepare_part_of_data(train_x, max_len)
    valid_x, valid_mask = prepare_part_of_data(valid_x, max_len)
    test_x, test_mask = prepare_part_of_data(test_x, max_len)

    return (train_x, valid_x, test_x), (train_mask, valid_mask, test_mask)

def prepare_part_of_data(group_x, max_len):
    long_xs = []
    masks = []

    for i, x in enumerate(group_x):
        long_x = np.zeros(max_len).astype('int32')
        mask = np.zeros(max_len).astype('int32')

        length = min(len(x), max_len)
        long_x[:length] = x[:length]

        #long_x[:len(x)] = x

        mask[:length] = 1
        #mask[:len(x)] = 1

        long_xs.append(long_x)
        masks.append(mask)

    return long_xs, masks



def load_embeddings(params, words2idx):
    # load word vectors
    initial_embeddings = None
    vocsize = len(words2idx.keys())
    if params['initialize_word_vectors']:
        if params['vectors'] == 'default_word2vec_300':
            # standard word2vec
            print "Loading standard word2vec vectors"
            vector_file = defines.word2vec_vectors_filename
            vectors = gensim.models.Word2Vec.load_word2vec_format(vector_file, binary=True)
            #vector_file = fh.make_filename(defines.vectors_dir, 'default_word2vec_300', 'csv')
            #vectors = fh.read_json(vector_file)
        else:
            # my word2vec vectors
            print "Loading custom word2vec vectors"
            vector_file = fh.make_filename(defines.vectors_dir, params['vectors'], 'bin')
            vectors = gensim.models.Word2Vec.load(vector_file)

        print "Setting up initial embeddings"

        missing_count = 0
        total_count = 0
        total_emb_dims = params['word2vec_dim']
        if params['add_OOV_dim']:
            total_emb_dims += 1
        initial_embeddings = np.zeros([vocsize, total_emb_dims], dtype=float)
        for w in words2idx.keys():
            i = words2idx[w]
            total_count += 1
            if w in vectors:
                initial_embeddings[i, :params['word2vec_dim']] = vectors[w]
            # create a separate orthogonal dimension for OOV
            elif w == '__OOV__' and params['add_OOV_dim']:
                initial_embeddings[i, params['word2vec_dim']] = 1
            else:
                #print "no vector for", w
                missing_count += 1
                initial_embeddings[i, :params['word2vec_dim']] = 0.05 * \
                    np.random.uniform(-1.0, 1.0, (1, params['word2vec_dim']))

    print "total words =", total_count
    print "total words missing =", missing_count
    return initial_embeddings


def write_predictions(datasets, test_fold, dev_fold, predictions, items, output_dir):
    true_labels = labels.get_dataset_labels(datasets[0])
    predictions_df = pd.DataFrame(np.zeros([len(items), len(true_labels.columns)]),
                                  index=items, columns=true_labels.columns)

    pps = []
    f1s = []
    for i, item in enumerate(items):
        predictions_df.loc[item] = predictions[i]
    count = 0
    for d in datasets:
        test_items = ds.get_test_documents(d, test_fold)
        true_labels = labels.get_dataset_labels(d)
        print count, len(test_items)
        output_df = predictions_df.loc[test_items]
        #output_df = pd.DataFrame(np.zeros([len(items), len(true_labels.columns)]),
        #                         index=items, columns=true_labels.columns)
        #for i, item in enumerate(items):
        #    output_df.loc[item] = predictions[count+i]
        output_filename = fh.make_filename(output_dir, fh.get_basename(d), 'csv')
        output_df.to_csv(output_filename)
        f1, pp = evaluation.calc_macro_mean_f1_pp(true_labels.loc[test_items], output_df)
        print d, f1, pp
        #count += len(items)
        f1s.append(f1)
        pps.append(pp)

    output_filename = fh.make_filename(output_dir, 'test_summary_f1', 'csv')
    question_f1s_df = pd.DataFrame(np.array(f1s).reshape([1, len(datasets)]),
                                   index=['macro f1'], columns=datasets)
    question_f1s_df.to_csv(output_filename)
    output_filename = fh.make_filename(output_dir, 'test_summary_pp', 'csv')
    question_pps_df = pd.DataFrame(np.array(pps).reshape([1, len(datasets)]),
                                   index=['percent perfect'], columns=datasets)
    question_pps_df.to_csv(output_filename)



def calc_mean_f1(predictions, true):
    mean_f1 = 0.0
    for i in range(len(predictions)):
        pred_item = predictions[i]
        if len(pred_item.shape) == 2:
            pred_item = pred_item[0, :]
        y_item = true[i]
        #print pred_item.shape
        #print y_item.shape
        pp = np.sum(pred_item)
        tpp = np.dot(y_item, pred_item)
        tp = np.sum(y_item)

        prec = tpp / max(1.0, float(pp))
        recall = tpp / max(1.0, float(tp))
        f1_item = max(0.0, 2 * prec * recall / (prec + recall))
        mean_f1 += f1_item
    mean_f1 /= float(len(predictions))
    return mean_f1


def save_predictions(p, g, w, filename):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'True\tPredicted\n'
        out += str(sl) + '\t' + str(max(sp)) + '\n'
        out += 'Word\tPrediction\n'
        for wp, w in zip(sp, sw):
            out += str(w) + '\t' + str(wp) + '\n'
        out += '\n'

    with codecs.open(filename, 'w') as f:
        f.writelines(out)

