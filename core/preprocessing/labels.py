import os
import json
import codecs
from optparse import OptionParser

import pandas as pd

from ..util import file_handling as fh, defines
import data_splitting as ds


def make_label_metaindex():
    input_filename = os.path.join('.', 'codes.json')
    with codecs.open(input_filename, 'r') as input_file:
        codes = json.load(input_file)

    label_index = {}

    for question in codes.keys():
        for mapping in codes[question]:
            orig = mapping[0]
            mine = int(mapping[1])
            if mine not in label_index.keys():
                label_index[mine] = {}
            label_index[mine][question] = orig

    return label_index


def get_dataset_labels(dataset):
    input_dir = defines.data_raw_labels_dir
    input_filename = fh.make_filename(input_dir, fh.get_basename(dataset), 'csv')
    label_data = pd.read_csv(input_filename, header=0, index_col=0)
    return label_data


def get_labels(datasets):
    all_labels = []
    for f in datasets:
        labels = get_dataset_labels(f)
        all_labels.append(labels)

    return pd.concat(all_labels, axis=0)


def get_powerset_labels(datasets):

    powerset_df = pd.DataFrame(columns=['powerset_index'])

    all_labels = get_labels(datasets)
    powerset_index = []

    for i in all_labels.index:
        key = str(list(all_labels.loc[i, :]))
        if key not in powerset_index:
            powerset_index.append(key)
        index = powerset_index.index(key)
        powerset_df.loc[i] = index

    return powerset_df, powerset_index


def output_label_counts(datasets):
    label_dict = {}
    all_labels = get_labels(datasets)
    n, p = all_labels.shape
    for i in all_labels.index:
        key = str(list(all_labels.loc[i, :]))
        label_dict[key] = label_dict.get(key, 0) + 1

    gt1 = 0
    gt2 = 0
    for k in label_dict:
        if label_dict[k] > 1:
            gt1 += 1
        if label_dict[k] > 2:
            gt2 += 1

    print "total keys = ", len(label_dict.keys())
    print "greater than 1 =", gt1
    print "greater than 2 =", gt2



def get_groups(group_file):
    groups = []
    lines = fh.read_text(group_file)
    for line in lines:
        if len(line) > 0:
            groups.append(line.split())
    return groups


def main():
    labels = get_labels(['Republican-Dislikes'])
    train, dev, test = ds.get_all_splits(0, None)

    df = pd.DataFrame(columns=labels.columns)
    df.loc['n_train'] = labels.loc[train, :].sum(axis=0)
    df.loc['n_test'] = labels.loc[test, :].sum(axis=0)
    df.to_csv('/Users/dcard/Desktop/temp/counts.csv')

if __name__ == '__main__':
    main()

