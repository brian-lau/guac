
from optparse import OptionParser

from ..util import defines
from ..old_extractors import features
from ..util import file_handling as fh

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
    return 'frames'


def get_prefix():
    return '_fr_'


def collect_semafor_pos_unifram_tokens(input_filename, prefix=''):

    sent_index = fh.read_json(input_filename)
    if prefix == '':
        prefix = get_prefix()

    basename = fh.get_basename(input_filename)
    frames_filename = fh.make_filename(defines.data_semafor_dir, basename, 'fes')
    frames = fh.read_text(frames_filename)

    data = {}
    values = set(sent_index.values())
    for v in values:
        data[v] = []

    # read all sentences, except for the last (blank) line
    for index, sent in enumerate(frames[:-1]):
        print index, sent
        parts = sent.split('\t')
        frame = get_prefix() + parts[2]
        sent = int(parts[6])
        key = sent_index[sent]
        data[key].append(frame)

    return data


if __name__ == '__main__':
    main()
