import re
from optparse import OptionParser

from old_extractors import features
from old_extractors.features import tokenizer
from util import file_handling as fh, defines


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('-d', dest='dataset', default='',
    #                  help='Dataset to process; (if not specified, all files in raw data directory will be used)')
    parser.add_option('-n', dest='length', default=3,
                      help='max sentence length; default=%default')
    (options, args) = parser.parse_args()

    #if options.dataset != '':
    #    input_dir = defines.data_raw_labels_dir
    #    filename = fh.make_filename(input_dir, options.dataset, 'csv')
    #    files = [filename]
    #else:
    input_dir = defines.data_raw_labels_dir
    files = fh.ls(input_dir, '*.csv')

    n = int(options.length)

    print "Extracting ngram tokens:"
    data = {}
    for f in files:
        print f
        extract_ngram_tokens(f, n, data)

    output_dir = fh.makedirs(defines.data_token_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(n), 'json')
    fh.write_to_json(data, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(n), get_prefix(n), filename=get_feature_name(n)+'_default',
                                     min_doc_threshold=2, binarize=True, feature_type='tokens')


def get_feature_name(n):
    return 'short_sentences_' + str(n)


def get_prefix(n):
    return '_ss' + str(n) + '_'


def extract_ngram_tokens(input_filename, n, data, prefix=''):

    Y = fh.read_csv(input_filename)
    rids = Y.index

    responses = fh.read_json(defines.data_raw_text_file)

    if prefix == '':
        prefix = get_prefix(n)

    for rid in rids:
        text = responses[rid].lower()
        text = text.lstrip()
        text = text.rstrip()
        text = text.lstrip('/')
        text = re.sub('<', '', text)
        text = re.sub('>', '', text)
        text = re.sub('-', ' - ', text)
        text = re.sub('_', ' - ', text)
        tokens = []
        paragraphs = re.split('[/\\\\]', text)
        paragraphs = [p for p in paragraphs if p != '']
        count = 0
        for p in paragraphs:
            count += 1
            sentences = tokenizer.split_sentences(p)
            for s in sentences:
                sent_tokens = tokenizer.make_ngrams(s, n=1)
                if len(sent_tokens) <= n:
                    sent = s.lstrip()
                    sent = sent.rstrip('. ')
                    sent = re.sub(' ', '_', sent)
                    tokens = tokens + [sent]

        tokens = [prefix + t for t in tokens]
        data[rid] = tokens


if __name__ == '__main__':
    main()
