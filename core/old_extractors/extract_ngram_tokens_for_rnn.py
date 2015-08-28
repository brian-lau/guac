import re
from collections import Counter
from optparse import OptionParser

from ..old_extractors import features
from ..feature_extractors import vocabulary, tokenizer
from ..util import file_handling as fh, defines
from ..util import data_splitting as ds


def main():
    # Handle input options and arguments
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-g', dest='group_file', default='',
                      help='List of datasets to process; (if not specified, all files in raw data directory will be used)')
    #parser.add_option('-n', dest='ngrams', default=1,
    #                  help='n for ngrams; default=%default')
    (options, args) = parser.parse_args()

    #if options.dataset != '':
    #    input_dir = defines.data_raw_labels_dir
    #    filename = fh.make_filename(input_dir, options.dataset, 'csv')
    #    files = [filename]
    #else:
    input_dir = defines.data_raw_labels_dir
    if options.group_file == '':
        files = fh.ls(input_dir, '*.csv')
    else:
        groups = fh.read_text(options.group_file)
        files = []
        for g in groups:
            datasets = g.split()
            for d in datasets:
                files.append(fh.make_filename(input_dir, d, 'csv'))

    print "Extracting ngram tokens:"
    data = {}
    train_items = []
    for f in files:
        print f
        extract_ngram_tokens(f, data)
        train_items.extend(ds.get_train_documents(f, 0, 0))

    output_dir = fh.makedirs(defines.data_token_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(), 'json')
    fh.write_to_json(data, output_filename)

    print "Counting tokens"
    token_counts = Counter()
    token_doc_counts = Counter()

    for rid in data.keys():
        if rid in train_items:
            token_counts.update(data[rid])
            token_doc_counts.update(set(data[rid]))

    print "Vocabulary size before pruning:", len(token_counts)

    min_threshold = 1
    valid_tokens = [t for (t, c) in token_doc_counts.items() if c >= min_threshold]

    print "Making vocabulary"
    vocab = vocabulary.Vocab('', tokens_to_add=valid_tokens)
    print "Vocabulary size after pruning:", len(vocab)

    print "Saving vocabulary"
    output_filename = fh.make_filename(output_dir, get_feature_name() + '_vocab', 'json')
    fh.write_to_json(vocab.token2index, output_filename)

    print "Extracting indices"
    indices = {}
    for rid in data.keys():
        indices[rid] = vocab.get_indices(data[rid]).tolist()
    print "Saving indices"
    output_filename = fh.make_filename(output_dir, get_feature_name() + '_indices', 'json')
    fh.write_to_json(indices, output_filename)

    # write default function definition
    features.make_feature_definition(get_feature_name(), get_prefix(), filename=get_feature_name()+'_default',
                                     min_doc_threshold=1, binarize=True, feature_type='tokens')


def get_feature_name(n=1):
    return 'ngrams_' + str(n) + '_rnn'


def get_prefix(n=1):
    return ''


def extract_ngram_tokens(input_filename, data, prefix=''):

    Y = fh.read_csv(input_filename)
    rids = Y.index

    responses = fh.read_json(defines.data_raw_text_file)

    if prefix == '':
        prefix = get_prefix()

    for rid in rids:
        text = responses[rid].lower()
        text = text.lstrip()
        text = text.rstrip()
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
                #sent_tokens = [t for t in sent_tokens if re.search('[a-z.?,!\'"`]', t)]
                sent_tokens = [t.rstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                sent_tokens = [t.lstrip('`"\'') if re.search('[a-z]', t) else t for t in sent_tokens]
                #sent_tokens = sent_tokens + ['__ENDS__']
                tokens = tokens + sent_tokens
        tokens = [prefix + t for t in tokens]
        data[rid] = tokens



if __name__ == '__main__':
    main()
