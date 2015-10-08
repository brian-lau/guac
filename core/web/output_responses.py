import codecs

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

from . import common

def output_responses():
    dataset = 'Democrat-Likes'

    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    rnn_dir = fh.makedirs(defines.exp_dir, 'rnn', 'bayes_opt_rnn_LSTM_reuse_mod_34_rerun', 'fold0', 'responses')

    text_file = defines.data_normalized_text_file
    text = fh.read_json(text_file)

    true = labels.get_labels([dataset])
    test_items = ds.get_test_documents(dataset, 0)

    for i in test_items:

        true_i = true.loc[i]
        rnn_file = fh.make_filename(rnn_dir, i, 'csv')
        rnn_vals = pd.read_csv(rnn_file, header=-1)
        rnn_vals.columns = true.columns

        output_filename = fh.make_filename(output_dir, i, 'html')
        with codecs.open(output_filename, 'w') as output_file:

            output_file.write(common.make_header(i))
            output_file.write(common.make_body_start())
            table_header = ['Label', 'Human coding'] + text[i].split()
            n_cols = len(table_header)
            output_file.write(common.make_table_start(n_cols, sortable=False))
            output_file.write(common.make_table_header(table_header))
            for code in true.columns:
                colours = 2*[str((0, 0, 0))]
                vals = [int(255 - (v*255)) for v in rnn_vals[code]]
                colours += [(v, 255, v) for v in vals]
                row = [code, str(true_i[code])] + text[i].split()
                output_file.write(common.make_table_row(row, colours=colours))
            output_file.write(common.make_table_end())

            output_file.write(common.make_body_end())
            output_file.write(common.make_footer())

def main():
    output_responses()

if __name__ == '__main__':
    main()
