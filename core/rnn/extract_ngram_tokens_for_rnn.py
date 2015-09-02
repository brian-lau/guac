import re
from collections import Counter
from optparse import OptionParser

from ..feature_extractors import vocabulary, tokenizer
from ..util import file_handling as fh, defines


def prepare_data_for_rnn(datasets, min_threshold=1, n=1):
    input_filename = defines.data_normalized_text_file
    responses = fh.read_json(input_filename)

    print "Extracting ngram tokens:"
    data = {}
    for key in responses.keys():
        extract_ngram_tokens(key, responses[key], data, n)

    output_dir = fh.makedirs(defines.data_rnn_dir)
    output_filename = fh.make_filename(output_dir, get_feature_name(n, min_threshold), 'json')
    fh.write_to_json(data, output_filename)

    print "Counting tokens"
    token_counts = Counter()
    token_doc_counts = Counter()

    #exclude_questions = ['Current-Industry', 'Current-Occupation', 'Past-Industry', 'Past-Occupation',
    #                     'drld-synthetic']

    for rid in data.keys():
        parts = rid.split('_')
        if parts[0] in datasets:
            token_counts.update(data[rid])
            token_doc_counts.update(set(data[rid]))

    print "Vocabulary size before pruning:", len(token_counts)

    valid_tokens = [t for (t, c) in token_doc_counts.items() if c >= min_threshold]

    print "Making vocabulary"
    vocab = vocabulary.Vocab('', tokens_to_add=valid_tokens)
    print "Vocabulary size after pruning:", len(vocab)

    print "Saving vocabulary"
    output_filename = fh.make_filename(output_dir, get_feature_name(n, min_threshold) + '_vocab', 'json')
    fh.write_to_json(vocab.token2index, output_filename)

    print "Extracting indices"
    indices = {}
    for rid in data.keys():
        indices[rid] = vocab.get_indices(data[rid]).tolist()
    print "Saving indices"
    output_filename = fh.make_filename(output_dir, get_feature_name(n, min_threshold) + '_indices', 'json')
    fh.write_to_json(indices, output_filename)



def get_feature_name(n, m):
    return 'ngrams_n' + str(n) + '_m' + str(m) + '_rnn'


def extract_ngram_tokens(key, text, all_tokens, n):

    text = text.lstrip()
    text = text.rstrip()
    tokens = []
    sentences = tokenizer.split_sentences(text)
    for s in sentences:
        sent_tokens = tokenizer.make_ngrams(s, n=n)
        # remove single quotes from words
        sent_tokens = [t.rstrip('\'') if re.search('[a-z]', t) else t for t in sent_tokens]
        sent_tokens = [t.lstrip('\'') if re.search('[a-z]', t) else t for t in sent_tokens]
        tokens = tokens + sent_tokens

    all_tokens[key] = tokens



def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='min_doc_thresh', default=1,
                      help='Minimum doc count threshold for inclusion of words; default=%default')
    (options, args) = parser.parse_args()

    m = int(options.min_doc_thresh)
    prepare_data_for_rnn(min_threshold=m)

if __name__ == '__main__':
    main()
