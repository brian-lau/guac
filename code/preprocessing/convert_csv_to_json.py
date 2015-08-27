from optparse import OptionParser

import pandas as pd
import numpy as np

from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('-d', dest='dataset', default='',
    #                  help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    (options, args) = parser.parse_args()

    # process all the (given) data files
    convert_data()

    #combine_responses()


def convert_data():
    input_dir = defines.data_raw_csv_dir
    files = fh.ls(input_dir, '*.csv')

    text_dir = fh.makedirs(defines.data_raw_text_dir)
    label_dir = fh.makedirs(defines.data_raw_labels_dir)

    responses = {}

    for f in files:
        # get basename of input file
        basename = fh.get_basename(f)
        print basename

        # read the data into a dataframe
        df = pd.read_csv(f, header=0, index_col=0)
        nRows, nCols = df.shape
        print nRows, nCols
        print df.index[0:10]

        index = df.index
        #print index
        # make the indices unique by prefixing with the question name
        index = [basename + '_' + str(i) for i in index]
        df.index = index

        # pull out the non-label columns
        [nRows, nCols] = df.shape
        print nRows, nCols

        col_sel = range(nCols)
        col_sel.pop(0)
        col_sel.pop(-1)

        # make a new dataframe of just the labels and write it to a file
        Y = df[col_sel]
        Y = pd.DataFrame(np.array(Y > 0, dtype=int), index=Y.index, columns=Y.columns)
        Y.to_csv(fh.make_filename(label_dir, basename, 'csv'))

        # pull out the text and add it to a dictionary of all responses
        X = df['Interviewer transcript']

        for x in X.iteritems():
            responses[x[0]] = x[1]

    # write the texts to a file
    output_filename = defines.data_raw_text_file
    fh.write_to_json(responses, output_filename)


# concatenate all of the text from each respondent for possible future use
def combine_responses():
    input_dir = defines.data_raw_text_dir
    output_dir = fh.makedirs(defines.data_raw_concat_dir)

    files = fh.ls(input_dir, '*.json')

    all_text = {}

    for f in files:
        text = fh.read_json(f)
        keys = text.keys()
        for k in keys:
            if all_text.has_key(k):
                all_text[k] += ' ; ' + text[k]
            else:
                all_text[k] = text[k]

    output_filename = fh.make_filename(output_dir, 'all_text', 'json')
    fh.write_to_json(all_text, output_filename)


if __name__ == '__main__':
    main()
