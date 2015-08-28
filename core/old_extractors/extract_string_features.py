import re
import sys
from optparse import OptionParser

from ..old_extractors import features
from ..old_extractors.features import tokenizer
from ..util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog string_features.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('-d', dest='dataset', default='',
    #                  help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    (options, args) = parser.parse_args()

    if len(args) < 1:
        sys.exit("Please provide a list of string features to extract (per dataset)")

    string_features_filename = args[0]
    string_features = fh.read_json(string_features_filename)

    input_dir = defines.data_raw_labels_dir
    files = fh.ls(input_dir, '*.csv')

    print "Extracting string tokens:"
    data = {}
    for f in files:
        print f
        extract_string_tokens(f, data, string_features)

    output_dir = fh.makedirs(defines.data_token_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(), 'json')
    fh.write_to_json(data, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(), get_prefix(), filename=get_feature_name()+'_default',
                                     min_doc_threshold=1, binarize=True, feature_type='tokens')


def get_feature_name():
    return 'hand_crafted_strings'


def get_prefix():
    return '_s_'


def extract_string_tokens(input_filename, data, string_features):

    Y = fh.read_csv(input_filename)
    rids = Y.index

    dataset = fh.get_basename(input_filename)
    prefix = get_prefix()

    if dataset in string_features:
        strings = string_features[dataset]
        responses = fh.read_json(defines.data_raw_text_file)
        for rid in rids:
            text = responses[rid].lower()
            tokens = []
            paragraphs = text.split('/')
            paragraphs = [p for p in paragraphs if p != '']
            for p in paragraphs:
                sentences = tokenizer.split_sentences(p)
                for s in sentences:
                    tokens.extend([t for t in strings if t in s])

            tokens = [prefix + re.sub(' ', '_', t) for t in tokens]
            data[rid] = tokens


if __name__ == '__main__':
    main()
