__author__ = 'dcard'
import re
import codecs

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

import html
import common
from codes import code_names

def output_response_index():
    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    output_filename = fh.make_filename(output_dir, 'index_responses', 'html')
    datasets = ['Democrat-Dislikes', 'Democrat-Likes', 'Republican-Dislikes', 'Republican-Likes']

    text_file_dir = fh.makedirs(defines.data_dir, 'rnn')
    text = fh.read_json(fh.make_filename(text_file_dir, 'ngrams_n1_m1_rnn', 'json'))

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(html.make_header('Democrats vs Republicans'))
        output_file.write(html.make_body_start())

        output_file.write(common.make_masthead(0))

        for dataset in datasets:
            true = labels.get_labels([dataset])
            all_items = ds.get_all_documents(dataset)
            train_items = ds.get_train_documents(dataset, 0, 0)
            dev_items = ds.get_dev_documents(dataset, 0, 0)
            test_items = ds.get_test_documents(dataset, 0)

            output_file.write(html.make_heading(dataset, align='center'))

            table_header = ['Response', 'Split', 'Snippet']
            col_widths = [130, 80, 800]
            output_file.write(html.make_table_start(col_widths=col_widths, style='sortable'))
            output_file.write(html.make_table_header(table_header))

            for subset in [train_items, dev_items, test_items]:
                subset.sort()
                for item in subset:
                    if item in train_items:
                        split = 'train'
                    elif item in dev_items:
                        split = 'dev'
                    else:
                        split = 'test'

                    words = text[item]
                    response = ' '.join(words)
                    if len(response) > 100:
                        response = response[:100] + '. . .'
                    num = item.split('_')[1]
                    link = html.make_link(item + '.html', num, new_window=True)
                    link2 = html.make_link(item + '.html', response, new_window=True)
                    row = [link, split, link2]
                    output_file.write(html.make_table_row(row))

            output_file.write(html.make_table_end())

            output_file.write(html.make_body_end())
            output_file.write(html.make_footer())

def main():
    output_response_index()

if __name__ == '__main__':
    main()
