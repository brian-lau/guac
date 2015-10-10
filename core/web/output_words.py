__author__ = 'dcard'

import re
import codecs

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

import html
from codes import code_names

def output_responses():
    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    blm_dir = fh.makedirs(defines.exp_dir, 'Democrat-Dislikes_Democrat-Likes_Republican-Dislikes_Republican-Likes', 'test_fold_0', 'L1LR_all_groups_a0', 'models')

    text_file_dir = fh.makedirs(defines.data_dir, 'rnn')
    text = fh.read_json(fh.make_filename(text_file_dir, 'ngrams_n1_m1_rnn', 'json'))
    vocab = fh.read_json(fh.make_filename(text_file_dir, 'ngrams_n1_m1_rnn_vocab', 'json'))

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

    for word in word_list:
        output_filename = fh.make_filename(output_dir, 'wordtype_' + word, 'html')
        with codecs.open(output_filename, 'w') as output_file:

            output_file.write(html.make_header(word))
            output_file.write(html.make_body_start())
            output_file.write(html.make_heading('Word: ' + word, align='center'))

            table_header = ['Label', 'Value', 'Scaled']
            output_file.write(html.make_table_start(style='sortable'))
            output_file.write(html.make_table_header(table_header))
            for code_index, code in enumerate(true.columns):
                # load coefficients from unigram model
                model_filename = fh.make_filename(blm_dir, re.sub(' ', '_', code), 'json')
                model = fh.read_json(model_filename)
                intercept = float(model.get('intercept', 1.0))

                cmax = 255
                if 'coefs' in model:
                    coefs = dict(model['coefs'])
                    colours = [str((0, 0, 0))]*2
                    coef = coefs.get('_n1_' + word, 0.0)
                    scaled_coef = coef/abs(intercept)
                    val = int(cmax - (min(1, abs(scaled_coef))*cmax))
                    if coef > 0:
                        colours += [(val, val, cmax)]
                    else:
                        colours += [(cmax, val, val)]
                else:
                    coef = 0.0
                    colours = [str((0, 0, 0)), str((0, 0, 0)), str((cmax, cmax, cmax))]

                code_name = code_names[code_index]
                link = html.make_link('label_' + html.replace_chars(code_name) + '.html', code_name)
                row = [link, '{:0.2f}'.format(coef), word]
                output_file.write(html.make_table_row(row, colours=colours))

            output_file.write(html.make_table_end())

            output_file.write(html.make_body_end())
            output_file.write(html.make_footer())

def main():
    output_responses()

if __name__ == '__main__':
    main()
