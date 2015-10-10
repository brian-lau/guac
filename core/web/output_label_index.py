import re
import codecs

from string import ascii_lowercase


from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels

import html
import common
from codes import code_names

def output_label_index():

    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    output_filename = fh.make_filename(output_dir, 'index_labels', 'html')

    true = labels.get_labels(['Democrat-Dislikes', 'Democrat-Likes', 'Republican-Dislikes', 'Republican-Likes'])

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(html.make_header('Labels'))
        output_file.write(html.make_body_start())
        output_file.write(common.make_masthead(1))
        output_file.write(html.make_heading('Labels', align='center'))

        table_header = ['Label']
        output_file.write(html.make_table_start(style='sortable'))
        output_file.write(html.make_table_header(table_header))

        for index, code in enumerate(true.columns):
            code_name = code_names[index]
            link = html.make_link('label_' + html.replace_chars(code_name) + '.html', code_name)
            row = [link]
            output_file.write(html.make_table_row(row))

        output_file.write(html.make_table_end())

        output_file.write(html.make_body_end())
        output_file.write(html.make_footer())

def main():
    output_label_index()

if __name__ == '__main__':
    main()
