
from optparse import OptionParser

from ..util import defines
from ..util import file_handling as fh
from ..old_extractors import features

def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dataset', default='',
                      help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    (options, args) = parser.parse_args()

    if options.dataset != '':
        input_dir = defines.data_raw_sents_dir
        filename = fh.make_filename(input_dir, options.dataset, 'json')
        files = [filename]
    else:
        input_dir = defines.data_raw_sents_dir
        files = fh.ls(input_dir, '*.json')

    print "Extracting ngram tokens:"
    for f in files:
        print f
        output_dir = fh.makedirs(defines.data_token_dir, get_feature_name())
        basename = fh.get_basename(f)
        data = collect_semafor_pos_unifram_tokens(f)
        output_filename = fh.make_filename(output_dir, basename, 'json')
        fh.write_to_json(data, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(), filename=get_feature_name()+'_default',
                                     min_doc_threshold=2, binarize=True)


def get_feature_name():
    return 'pos_unigram'


def get_prefix():
    return '_pn1_'


def collect_semafor_pos_unifram_tokens(input_filename, prefix=''):

    sent_index = fh.read_json(input_filename)
    if prefix == '':
        prefix = get_prefix()

    basename = fh.get_basename(input_filename)
    pos_filename = fh.make_filename(defines.data_semafor_dir, basename, 'pos')
    sents = fh.read_text(pos_filename)

    data = {}
    # read all sentences, except for the last (blank) line
    for index, sent in enumerate(sents[:-1]):
        print index, sent
        tokens = sent.split()
        tokens = [prefix + t for t in tokens]
        key = sent_index[index]
        if data.has_key(key):
            data[key] = data[key] + tokens
        else:
            data[key] = tokens

    return data


if __name__ == '__main__':
    main()
