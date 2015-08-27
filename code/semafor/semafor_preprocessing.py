import codecs
from optparse import OptionParser

import defines
from features import tokenizer
import file_handling as fh

def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dataset', default='',
                      help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    (options, args) = parser.parse_args()

    if options.dataset != '':
        input_dir = defines.data_raw_text_dir
        filename = fh.make_filename(input_dir, options.dataset, 'json')
        files = [filename]
    else:
        input_dir = defines.data_raw_text_dir
        files = fh.ls(input_dir, '*.json')

    # process all the (given) data files
    print "Converting .csv files:"
    for f in files:
        print f
        write_sentences(f)


def write_sentences(f):
    input_dir = defines.data_raw_text_dir
    output_dir = fh.makedirs(defines.data_raw_sents_dir)

    index = 0
    sent_index = {}
    responses = fh.read_json(f)
    keys = responses.keys()
    keys.sort()

    sentences_filename = fh.make_filename(output_dir, fh.get_basename(f), 'txt')
    index_filename = fh.make_filename(output_dir, fh.get_basename(f), 'json')
    with codecs.open(sentences_filename, 'w') as output_file:
        for k in keys:
            text = responses[k].lower()
            paragraphs = text.split('/')
            paragraphs = [p for p in paragraphs if p != '']
            for p in paragraphs:
                sentences = tokenizer.split_sentences(p)
                for sent in sentences:
                    if len(sent) > 0:
                        output_file.write(sent + '\n')
                        sent_index[index] = k
                        index += 1

    fh.write_to_json(sent_index, index_filename)


if __name__ == '__main__':
    main()
