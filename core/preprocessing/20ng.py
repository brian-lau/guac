from optparse import OptionParser
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from ..util import defines


def main():
    usage = "%prog subj0 subj1 "
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    make_heads_up_comparison(args[0], args[1])


def make_heads_up_comparison(subj0, subj1):

    text = {}

    cats = [subj0, subj1]
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)

    labels = pd.DataFrame(columns=[subj1])
    splits = pd.DataFrame(columns=['major_split', 'minor_split', 'conf_split'])

    j = 0
    k = 0
    n_dev_folds = 5
    n_conf_folds = 2

    for i, filename in enumerate(train['filenames']):
        basename = fh.get_basename(filename)
        target_name = train['target_names'][train['target'][i]]
        if target_name == subj0:
            labels.loc[basename, subj1] = 0
            text[basename] = train['data'][i]
        elif target_name == subj1:
            labels.loc[basename, subj1] = 1
            text[basename] = train['data'][i]
        splits.loc[basename, :] = [1, j, k]
        k += 1
        if k == n_conf_folds:
            k = 0
            j += 1
            if j == n_dev_folds:
                j = 0

    for i, filename in enumerate(test['filenames']):
        basename = fh.get_basename(filename)
        target_name = test['target_names'][test['target'][i]]
        if target_name == subj0:
            labels.loc[basename, subj1] = 0
            text[basename] = test['data'][i]
        elif target_name == subj1:
            labels.loc[basename, subj1] = 1
            text[basename] = test['data'][i]
        splits.loc[basename, :] = [0, j, k]
        k += 1
        if k == n_conf_folds:
            k = 0
            j += 1
            if j == n_dev_folds:
                j = 0

    print labels.shape
    print len(text.keys())

    output_dir = fh.makedirs(defines.base_dir, '20ng', 'raw', 'labels')
    output_filename = fh.make_filename(output_dir, 'religion', 'csv')
    labels.to_csv(output_filename)

    output_dir = fh.makedirs(defines.base_dir, '20ng', 'raw', 'text')
    output_filename = fh.make_filename(output_dir, 'text', 'json')
    fh.write_to_json(text, output_filename)

    output_dir = fh.makedirs(defines.base_dir, '20ng', 'subsets')
    output_filename = fh.make_filename(output_dir, 'religion', 'csv')
    splits.to_csv(output_filename)

    print "Done"



if __name__ == '__main__':
    main()
