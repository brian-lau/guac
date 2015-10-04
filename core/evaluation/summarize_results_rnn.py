import os
from optparse import OptionParser
import glob

import numpy as np
import pandas as pd

from core.util import defines
from core.util import file_handling as fh




def main():

    usage = "%prog results_file"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    results_file = args[0]

    results = pd.DataFrame(columns=('masked', 'test', 'valid', 'dir'))

    lines = fh.read_text(results_file)

    for i, line in enumerate(lines):
        run_num = i

        if i > 0 and run_num < 42:
            parts = line.split()
            date = parts[0]
            time = parts[1]
            name = parts[2]
            masked = float(parts[3])
            test = float(parts[4])
            if test > 0:
                valid = parts[5][1:-1]
            else:
                valid = masked

            #results.loc[run_num, 'iteration'] = run_num
            results.loc[run_num, 'masked'] = masked
            results.loc[run_num, 'test'] = test
            results.loc[run_num, 'valid'] = valid
            results.loc[run_num, 'dir'] = name

    results.to_csv(results_file + 'results.csv', columns=results.columns)

    sorted = results.sort('masked')
    print sorted

    print "best by masked"
    print sorted.values[-1, :]

    print "best by valid"
    sorted = results.sort('valid')
    print sorted.values[-1, :]


if __name__ == '__main__':
    main()
