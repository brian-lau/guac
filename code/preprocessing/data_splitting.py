import csv
import codecs
import random
from optparse import OptionParser

import pandas as pd

from ..util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dataset', default='',
                      help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    parser.add_option('-t', dest='test_folds', default=10,
                      help='Number of test folds; default=%default')
    parser.add_option('-v', dest='dev_subfolds', default=5,
                      help='Number of dev subfolds; default=%default')
    parser.add_option('-s', dest='seed', default=42,
                      help='Seed for randomization; default=%default')
    (options, args) = parser.parse_args()

    if options.dataset:
        input_dir = defines.data_raw_labels_dir
        filename = fh.make_filename(input_dir, options.dataset, 'csv')
        files = [filename]
    else:
        input_dir = defines.data_raw_labels_dir
        files = fh.ls(input_dir, '*.csv')

    seed = int(options.seed)
    random.seed(seed)

    t = int(options.test_folds)
    v = int(options.dev_subfolds)
    print "Extracting ngram tokens:"
    for f in files:
        print f
        assign_data_split(f, t, v)


def assign_data_split(input_filename, t, v):
    output_dir = fh.makedirs(defines.data_subsets_dir)

    # get basename of input file
    basename = fh.get_basename(input_filename)

    # read the data into a dataframe
    df = pd.read_csv(input_filename, header=0, index_col=0)
    articles = df.index.tolist()

    random.shuffle(articles)

    output_filename = fh.make_filename(output_dir, basename, 'csv')
    with codecs.open(output_filename, 'w') as output_file:
        writer = csv.writer(output_file)
        header = ['rid', 'major_split', 'minor_split']
        writer.writerow(header)
        i = 0
        j = 0
        for article in articles:
            writer.writerow([article, i, j])
            i += 1
            if i >= t:
                i = 0
                j += 1
                if j >= v:
                    j = 0

def get_all_splits(test_fold, dev_subfold):
    label_files = fh.get_label_files()
    train = []
    dev = []
    test = []
    for f in label_files:
        train.extend(get_train_documents(f, test_fold=test_fold, dev_subfold=dev_subfold))
        dev.extend(get_dev_documents(f, test_fold=test_fold, dev_subfold=dev_subfold))
        test.extend(get_test_documents(f, test_fold=test_fold))
    return train, dev, test


def get_train_documents(input_filename, test_fold, dev_subfold):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    temp_df = df[df['major_split'] != test_fold]
    if dev_subfold is not None:
        train_df = temp_df[temp_df['minor_split'] != dev_subfold]
    else:
        train_df = temp_df
    train_list = train_df.index.tolist()
    return train_list


def get_dev_documents(input_filename, test_fold, dev_subfold):
    if dev_subfold is not None:
        input_dir = defines.data_subsets_dir
        filename = fh.make_filename(input_dir, input_filename, 'csv')
        df = pd.read_csv(filename, header=0, index_col=0)
        temp_df = df[df['major_split'] != test_fold]
        dev_df = temp_df[temp_df['minor_split'] == dev_subfold]
        dev_list = dev_df.index.tolist()
        return dev_list
    else:
        return []


def get_test_documents(input_filename, test_fold):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    test_df = df[df['major_split'] == test_fold]
    test_list = test_df.index.tolist()
    return test_list

def get_nontest_documents(input_filename, test_fold):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    test_df = df[df['major_split'] != test_fold]
    test_list = test_df.index.tolist()
    return test_list

def get_all_documents(input_filename):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    return df.index.tolist()

def get_n_dev_folds(input_filename):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    return df['minor_split'].max()+1


if __name__ == '__main__':
    main()
