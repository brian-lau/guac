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

def output_word_index():
    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    blm_dir = fh.makedirs(defines.exp_dir, 'Democrat-Dislikes_Democrat-Likes_Republican-Dislikes_Republican-Likes', 'test_fold_0', 'L1LR_all_groups_a0', 'models')

    true = labels.get_labels(['Democrat-Dislikes', 'Democrat-Likes', 'Republican-Dislikes', 'Republican-Likes'])

    word_list = set()
    for code_index, code in enumerate(true.columns):
        # load coefficients from unigram model
        model_filename = fh.make_filename(blm_dir, html.replace_chars(code), 'json')
        model = fh.read_json(model_filename)
        if 'coefs' in model:
            coefs = dict(model['coefs'])
            words = [word[4:] for word in coefs.keys()]
            word_list.update(words)
    word_list = list(word_list)
    word_list.sort()

    output_filename = fh.make_filename(output_dir, 'index_words', 'html')
    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(html.make_header('Words'))
        output_file.write(html.make_body_start())
        output_file.write(common.make_masthead(2))
        output_file.write(html.make_heading('Words', align='center'))

        table_header = ['Words']
        output_file.write(html.make_table_start(style='sortable'))
        output_file.write(html.make_table_header(table_header))
        for word in word_list:
            link = html.make_link('wordtype_' + html.replace_chars(word) + '.html', word)
            row = [link]
            output_file.write(html.make_table_row(row))

        output_file.write(html.make_table_end())

        output_file.write(html.make_body_end())
        output_file.write(html.make_footer())

def main():
    output_word_index()

if __name__ == '__main__':
    main()
