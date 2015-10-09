__author__ = 'dcard'
import re
import codecs

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

from . import common


def output_response_index():
    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    output_filename = fh.make_filename(output_dir, 'DRLD_index', 'html')
    datasets = ['Democrat-Dislikes', 'Democrat-Likes', 'Republican-Dislikes', 'Republican-Likes']

    text_file_dir = fh.makedirs(defines.data_dir, 'rnn')
    text = fh.read_json(fh.make_filename(text_file_dir, 'ngrams_n1_m1_rnn', 'json'))

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(common.make_header('D/R Dis/likes index'))
        output_file.write(common.make_body_start())

        for dataset in datasets:
            true = labels.get_labels([dataset])
            all_items = ds.get_all_documents(dataset)
            train_items = ds.get_train_documents(dataset, 0, 0)
            dev_items = ds.get_dev_documents(dataset, 0, 0)
            test_items = ds.get_test_documents(dataset, 0)

            output_file.write(common.make_heading(dataset))

            table_header = ['Response', 'Split', 'Snippet']
            col_widths = [100, 80, 500]
            output_file.write(common.make_table_start(col_widths=col_widths, style='sortable'))
            output_file.write(common.make_table_header(table_header))

            for item in all_items:
                if item in train_items:
                    split = 'train'
                elif item in dev_items:
                    split = 'dev'
                else:
                    split = 'test'

                words = text[item]
                if len(words) > 10:
                    words = words[:10] + ['...']
                num = item.split('_')[1]
                link = common.make_link(item + '.html', num)
                row = [link, split, ' '.join(words)]
                output_file.write(common.make_table_row(row))

            output_file.write(common.make_table_end())

            output_file.write(common.make_body_end())
            output_file.write(common.make_footer())

def main():
    output_response_index()

if __name__ == '__main__':
    main()
