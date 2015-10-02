import os
from optparse import OptionParser
import glob

import numpy as np
import pandas as pd

from core.util import defines
from core.util import file_handling as fh




def main():

    usage = "%prog exp_dir_test_fold_dir"
    parser = OptionParser(usage=usage)

    parser.add_option('-t', dest='test_fold', default=0,
                      help='Test fold; default=%default')

    (options, args) = parser.parse_args()
    test_fold = options.test_fold
    exp_dir = args[0]

    results = pd.DataFrame(columns=('masked', 'test', 'valid', 'dir'))

    run_dirs = glob.glob(os.path.join(exp_dir, 'bayes*reuse*'))
    for i, dir in enumerate(run_dirs):
        run_num = int(fh.get_basename(dir).split('_')[-1])

        if run_num < 41:
            results_dir = os.path.join(dir, 'results')
            test_file = fh.make_filename(results_dir, 'test_macro_f1', 'csv')
            valid_file = fh.make_filename(results_dir, 'valid_cv_macro_f1', 'csv')
            masked_valid_file = fh.make_filename(results_dir, 'masked_valid_cv_macro_f1', 'csv')

            try:
                test = pd.read_csv(test_file, header=False, index_col=0)
                valid = pd.read_csv(valid_file, header=False, index_col=0)
                masked_valid = pd.read_csv(masked_valid_file, header=False, index_col=0)

                #results.loc[run_num, 'iteration'] = run_num
                results.loc[run_num, 'masked'] = masked_valid['overall'].mean()
                results.loc[run_num, 'test'] = test['overall'].mean()
                results.loc[run_num, 'valid'] = valid['overall'].mean()
                results.loc[run_num, 'dir'] = fh.get_basename(dir)
            except:
                continue

    results.to_csv(fh.make_filename(exp_dir, 'summary', 'csv'))

    sorted = results.sort('masked')
    print sorted

    print "best by masked"
    print sorted.values[-1, :]

    print "best by valid"
    sorted = results.sort('valid')
    print sorted.values[-1, :]




if __name__ == '__main__':
    main()
