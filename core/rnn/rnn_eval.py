import os
import pandas as pd
from optparse import OptionParser

from ..preprocessing import labels
from ..preprocessing import data_splitting as ds
from ..experiment import evaluation
from ..util import file_handling as fh
from ..util import defines

def main(param=None):

    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    folder = args[0]

    datasets = ['Democrat-Likes']

    # evaluate performance
    for d in datasets:
        test_items = ds.get_test_documents(d, 0)
        true_labels = labels.get_dataset_labels(d)

        report_header = evaluation.get_report_header(d, test_fold=0, dev_subfold=0)

        predictions_filename = fh.make_filename(folder, d, 'csv')
        predictions = pd.read_csv(predictions_filename, header=0, index_col=0)

        model_acc = evaluation.calc_task_wise_acc(true_labels.loc[test_items], predictions.loc[test_items], 'rnn')
        model_f1s = evaluation.calc_task_wise_f1s(true_labels.loc[test_items], predictions.loc[test_items], 'rnn')

        results = pd.concat([report_header, model_acc, model_f1s], axis=0)
        results_filename = fh.make_filename(folder, d + '_results', 'csv')
        results.to_csv(results_filename)


if __name__ == '__main__':
    main()