import csv
import codecs
import random
from optparse import OptionParser

import numpy as np
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


def assign_data_split(input_filename, t, v, calib_prop=0.5):
    output_dir = fh.makedirs(defines.data_subsets_dir)

    # get basename of input file
    basename = fh.get_basename(input_filename)

    # read the data into a dataframe
    df = pd.read_csv(input_filename, header=0, index_col=0)
    articles = df.index.tolist()

    random.shuffle(articles)

    rids = []
    majors = []
    minors = []
    calibs = []

    i = 0
    j = 0
    for article in articles:
        majors.append(i)
        minors.append(j)
        if np.random.rand() <= calib_prop:
            calibs.append(0)
        else:
            calibs.append(1)
        i += 1
        if i >= t:
            i = 0
            j += 1
            if j >= v:
                j = 0

    df_out = pd.DataFrame(index=articles, columns=['major_split', 'minor_split', 'calibration'])
    df_out['major_split'] = majors
    df_out['minor_split'] = minors
    df_out['calibration'] = calibs

    output_filename = fh.make_filename(output_dir, basename, 'csv')
    df_out.to_csv(output_filename)


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


def get_train_documents(input_filename, test_fold, dev_subfold, calibration=None):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    temp_df = df[df['major_split'] != test_fold]
    if dev_subfold is not None:
        train_df = temp_df[temp_df['minor_split'] != dev_subfold]
    else:
        train_df = temp_df
    if calibration is not None:
        train_df = train_df[train_df['calibration'] == int(calibration)]
    train_list = train_df.index.tolist()
    return train_list


def get_dev_documents(input_filename, test_fold, dev_subfold, calibration=None):
    if dev_subfold is not None:
        input_dir = defines.data_subsets_dir
        filename = fh.make_filename(input_dir, input_filename, 'csv')
        df = pd.read_csv(filename, header=0, index_col=0)
        temp_df = df[df['major_split'] != test_fold]
        dev_df = temp_df[temp_df['minor_split'] == dev_subfold]
        if calibration is not None:
            dev_df = dev_df[dev_df['calibration'] == int(calibration)]
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

def get_nontest_documents(input_filename, test_fold, calibration=None):
    input_dir = defines.data_subsets_dir
    filename = fh.make_filename(input_dir, input_filename, 'csv')
    df = pd.read_csv(filename, header=0, index_col=0)
    test_df = df[df['major_split'] != test_fold]
    if calibration is not None:
        test_df = test_df[test_df['calibration'] == int(calibration)]
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

def get_td_split_list(datasets, test_fold, calibration=None):
    df = get_tdt_splits(datasets)
    df = df[df['major_split'] != test_fold]
    if calibration is not None:
        df = df[df['calibration'] == int(calibration)]
    return df['minor_split'].tolist()

def get_tdt_splits(datasets, calibration=None):
    input_dir = defines.data_subsets_dir
    dataframes = []
    for f in datasets:
        filename = fh.make_filename(input_dir, f, 'csv')
        dataframes.append(pd.read_csv(filename, header=0, index_col=0))
    return pd.concat(dataframes, axis=0)


if __name__ == '__main__':
    main()
