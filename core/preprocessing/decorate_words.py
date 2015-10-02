import os
import re

from ..util import defines
from ..util import file_handling as fh

def decorate_words(output_filename, search_word1, search_word2):
    input_filename = defines.data_normalized_text_file
    data = fh.read_json(input_filename)

    decorated = {}
    for key, text in data.items():
        words = text.split()
        match = re.search(search_word1, key)
        if match is not None:
            words = [w + '_' + search_word1.upper() for w in words]
            decorated[key] = ' '.join(words)
        #else:
        #    words = [w + '_' + search_word.upper() + '-0' for w in words]
        match = re.search(search_word2, key)
        if match is not None:
            words = [w + '_' + search_word2.upper() for w in words]
            decorated[key] = ' '.join(words)

    fh.write_to_json(decorated, output_filename)


def main():

    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_dem', 'json')
    decorate_words(output_filename, 'Democrat', 'Republican')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_likes', 'json')
    decorate_words(output_filename, 'Likes', 'Dislikes')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_personal', 'json')
    decorate_words(output_filename, 'Personal', 'Political')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_mccain', 'json')
    decorate_words(output_filename, 'Obama', 'McCain')
    output_filename = fh.make_filename(defines.data_processed_text_dir, 'decorated_clinton', 'json')
    decorate_words(output_filename, 'Obama', 'Clinton')

if __name__ == '__main__':
    main()
