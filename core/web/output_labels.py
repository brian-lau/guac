import re
import codecs

from string import ascii_lowercase

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

import html
from codes import code_names

def output_label_pages():

    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    blm_dir = fh.makedirs(defines.exp_dir, 'Democrat-Dislikes_Democrat-Likes_Republican-Dislikes_Republican-Likes', 'test_fold_0', 'L1LR_all_groups_a0', 'models')
    true = labels.get_labels(['Democrat-Dislikes', 'Democrat-Likes', 'Republican-Dislikes', 'Republican-Likes'])

    for code_index, code in enumerate(true.columns):
        code_name = code_names[code_index]
        output_filename = fh.make_filename(output_dir, 'label_' + html.replace_chars(code_name), 'html')
        with codecs.open(output_filename, 'w') as output_file:

            output_file.write(html.make_header(code_name))
            output_file.write(html.make_body_start())
            output_file.write(html.make_heading('Label: ' + code_name, align='center'))

            table_header = ['Word', 'Value', 'Scaled']
            output_file.write(html.make_table_start(style='sortable'))
            output_file.write(html.make_table_header(table_header))

            model_filename = fh.make_filename(blm_dir, re.sub(' ', '_', code), 'json')
            model = fh.read_json(model_filename)
            intercept = float(model.get('intercept', 1.0))
            if 'coefs' in model:
                coefs = dict(model['coefs'])

                tokens = coefs.keys()
                tokens.sort()
                for token_index, token in enumerate(tokens):
                    cmax = 255
                    colours = [(0, 0, 0)]*2
                    word = token.split('_')[-1]
                    coef = coefs[token]
                    scaled_coef = coef/abs(intercept)
                    val = int(cmax - (min(1, abs(scaled_coef))*cmax))
                    if coef > 0:
                        colours += [(val, val, cmax)]
                    else:
                        colours += [(cmax, val, val)]

                    if len(word) > 0:
                        if word[0] not in ascii_lowercase:
                            word = '_' + word
                        link = html.make_link('wordtype_' + word + '.html', word)
                        row = [link, str('{:0.2f}'.format(coef)), word]
                        output_file.write(html.make_table_row(row, colours=colours))

            output_file.write(html.make_table_end())

            output_file.write(html.make_body_end())
            output_file.write(html.make_footer())

def main():
    output_label_pages()

if __name__ == '__main__':
    main()
__author__ = 'dcard'
