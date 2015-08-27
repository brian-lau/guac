from optparse import OptionParser

from old_extractors import features
from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    input_dir = defines.data_raw_labels_dir
    files = fh.ls(input_dir, '*.csv')

    print "Making basic question feature:"
    data = {}
    for f in files:
        print f
        extract_question_feature(f, data)

    output_dir = fh.makedirs(defines.data_token_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(), 'json')
    fh.write_to_json(data, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(), get_prefix(), filename=get_feature_name()+'_default',
                                     min_doc_threshold=1, binarize=True, feature_type='tokens')


def get_feature_name():
    return 'question'


def get_prefix():
    return '_q_'


def extract_question_feature(input_filename, data, prefix=''):

    Y = fh.read_csv(input_filename)
    rids = Y.index

    if prefix == '':
        prefix = get_prefix()

    for rid in rids:
        tokens = [prefix + fh.get_basename(input_filename)]
        data[rid] = tokens


if __name__ == '__main__':
    main()
