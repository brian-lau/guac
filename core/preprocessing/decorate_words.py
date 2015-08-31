import os
import re

from ..util import defines
from ..util import file_handling as fh

def decorate_words(output_filename, search_word):
    input_filename = defines.data_normalized_text_file
    data = fh.read_json(input_filename)

    decorated = {}
    for key, text in data.items():
        words = text.split()
        match = re.search(search_word, key)
        if match is not None:
            words = [w + '_' + search_word.upper() + '-1' for w in words]
        else:
            words = [w + '_' + search_word.upper() + '-0' for w in words]
        decorated[key] = ' '.join(words)

    fh.write_to_json(decorated, output_filename)


def main():

    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_dem', 'json')
    decorate_words(output_filename, 'Democrat')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_rep', 'json')
    decorate_words(output_filename, 'Republican')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_like', 'json')
    decorate_words(output_filename, 'Likes')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_dislike', 'json')
    decorate_words(output_filename, 'Dislikes')


if __name__ == '__main__':
    main()
