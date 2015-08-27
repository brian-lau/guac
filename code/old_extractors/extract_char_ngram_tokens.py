from optparse import OptionParser

from ..old_extractors import features
from ..util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog <filename>\n If filename not file specified, all files in raw data directory will be processed"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dataset', default='',
                      help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    parser.add_option('-n', dest='ngrams', default=3,
                      help='n for ngrams; default=%default')
    (options, args) = parser.parse_args()

    if options.dataset != '':
        input_dir = defines.data_raw_text_dir
        filename = fh.make_filename(input_dir, options.dataset, 'json')
        files = [filename]
    else:
        input_dir = defines.data_raw_labels_dir
        files = fh.ls(input_dir, '*.csv')

    n = int(options.ngrams)
    print "Extracting character ngrams:"
    data = {}
    for f in files:
        print f
        extract_char_ngrams(f, n, data)

    output_dir = fh.makedirs(defines.data_token_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(n), 'json')
    fh.write_to_json(data, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(n), filename=get_feature_name(n)+'_default',
                                     min_doc_threshold=3, binarize=True)


def get_feature_name(n):
    return 'char_ngrams_' + str(n)


def get_prefix(n):
    return '_cn' + str(n) + '_'


def extract_char_ngrams(input_filename, n, data, prefix=''):
    Y = fh.read_csv(input_filename)
    rids = Y.index

    responses = fh.read_json(defines.data_raw_text_file)

    if prefix == '':
        prefix = get_prefix(n)

    for rid in rids:
        text = responses[rid].lower()
        tokens = []
        nChars = len(text)
        for i in range(nChars-n+1):
            tokens.append(text[i:i+n])
        tokens = [prefix + t for t in tokens]
        data[rid] = tokens



if __name__ == '__main__':
    main()
