import os
import glob

import pandas as pd

from ..util import defines
from ..util import file_handling as fh

from ..preprocessing import labels


def main():
    exp_dir = defines.exp_dir
    exp_name = 'bayes_opt_LR_alphas_reuse'
    df = pd.DataFrame()

    basenames = ['test_acc.csv', 'test_micro_f1.csv', 'test_macro_f1.csv', 'test_pp.csv']
    rownames = ['model accuracy', 'model micro f1', 'model macro f1', 'model percent perfect']

    for i, basename in enumerate(basenames):
        rowname = rownames[i]
        files = glob.glob(os.path.join(exp_dir, '*', 'test_fold_0', exp_name, 'results', basename))
        gather_results(df, files, rowname)

    files = glob.glob(os.path.join(defines.data_raw_labels_dir, '*.csv'))
    for file in files:
        dataset = fh.get_basename(file)
        codes = labels.get_dataset_labels(dataset)
        if dataset in df.columns:
            df.loc['Number of responses', dataset] = codes.shape[0]
            df.loc['Number of labels', dataset] = codes.shape[1]

    output_dir = '/Users/dcard/Dropbox/CMU/DAP/results/'
    output_filename = fh.make_filename(output_dir, exp_name, 'csv')
    df.to_csv(output_filename)


def gather_results(df, files, rowname):
    for file in files:
        parts = file.split('/')
        dirname = parts[10]
        datasets = dirname.split('_')
        for d in datasets:
            results = pd.read_csv(file, header=False)
            df.loc[rowname, d] = results.loc[0, d]

if __name__ == '__main__':
    main()
