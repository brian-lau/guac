import os
from optparse import OptionParser
import glob

import numpy as np
import pandas as pd

from ..util import defines
from ..util import file_handling as fh




def main():

    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    exp_dir = args[0]

    results = pd.DataFrame(columns=('masked', 'test', 'valid', 'dir'))

    run_dirs = glob.glob(os.path.join(exp_dir, 'test_fold_0', 'bayes*'))
    for i, dir in enumerate(run_dirs):
        results_dir = os.path.join(dir, 'results')
        test_file = fh.make_filename(results_dir, 'test_macro_f1', 'csv')
        valid_file = fh.make_filename(results_dir, 'valid_cv_macro_f1', 'csv')
        masked_valid_file = fh.make_filename(results_dir, 'masked_valid_cv_macro_f1', 'csv')

        try:
            test = pd.read_csv(test_file, header=False, index_col=0)
            valid = pd.read_csv(valid_file, header=False, index_col=0)
            masked_valid = pd.read_csv(masked_valid_file, header=False, index_col=0)

            results.loc[i, 'masked'] = masked_valid['overall'].mean()
            results.loc[i, 'test'] = test['overall'].mean()
            results.loc[i, 'valid'] = valid['overall'].mean()
            results.loc[i, 'dir'] = fh.get_basename(dir)
        except:
            continue


    sorted = results.sort('masked')
    print sorted

    print "best by masked"
    print sorted.values[-1, :]

    print "best by valid"
    sorted = results.sort('valid')
    print sorted.values[-1, :]




if __name__ == '__main__':
    main()
