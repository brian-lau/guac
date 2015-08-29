
from ..util import defines
from ..util import file_handling as fh
from ..feature_extractors import normalizer


# Write each response to a separate file for parsing with CoreNLP
# (because each may contain multiple sentences)
def normalize_text():
    print "Normalizing text"
    input_filename = defines.data_raw_text_file
    output_filename = defines.data_normalized_text_file
    normalized_text = {}
    data = fh.read_json(input_filename)
    for key, line in data.items():
        line = normalizer.fix_basic_punctuation(line)
        normalized_text[key] = line

    fh.write_to_json(normalized_text, output_filename)


def main():
    normalize_text()

if __name__ == '__main__':
    main()
