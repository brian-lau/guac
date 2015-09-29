__author__ = 'dcard'

from optparse import OptionParser

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from ..preprocessing import labels

def main():

    usage = "%prog dataset filename.csv"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    find_most_errors(args[0], args[1])


def find_most_errors(dataset, filename):
    predicted = pd.read_csv(filename, index_col=0)
    df_labels = labels.get_dataset_labels(dataset)

    false_positives = []
    false_negatives = []
    for i in predicted.index:
        false_positives.append(np.sum(np.maximum(predicted.loc[i, :] - df_labels.loc[i, :], 0)))
        false_negatives.append(np.sum(np.maximum(df_labels.loc[i, :] - predicted.loc[i, :], 0)))

    print "false positives"
    order = np.argsort(false_positives)
    for i in range(1, 11):
        print false_positives[order[-i]], predicted.index[order[-i]]

    plt.hist(false_positives)
    plt.show()

    print "false negatives"
    order = np.argsort(false_negatives)
    for i in range(1, 11):
        print false_negatives[order[-i]], predicted.index[order[-i]]

    plt.hist(false_negatives)
    plt.show()

if __name__ == '__main__':
    main()

