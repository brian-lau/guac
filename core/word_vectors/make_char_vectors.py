from ..util import defines
from ..util import file_handling as fh


input_filename = defines.data_raw_text_file
data = fh.read_json(input_filename)

char_lists = {}
for key, line in data.items():
    #line = normalizer.fix_basic_punctuation(line)
    chars = list(line)
    char_lists[key] = chars

char_set = set()
for key, response in char_lists.items():
    chars = list(response)
    char_set.update(chars)

char_set = list(char_set)
char_set.sort()

char_index = dict(zip(char_set, range(len(char_set))))

char_indices = {}
for key, response in char_lists.items():
    char_indices[key] = [char_index[c] for c in response]

output_dir = defines.data_rnn_dir
output_filename = fh.make_filename(output_dir, 'chars_rnn', 'json')
fh.write_to_json(char_lists, output_filename)

output_filename = fh.make_filename(output_dir, 'chars_rnn_vocab', 'json')
fh.write_to_json(char_index, output_filename)

output_filename = fh.make_filename(output_dir, 'chars_rnn_indices', 'json')
fh.write_to_json(char_indices, output_filename)
