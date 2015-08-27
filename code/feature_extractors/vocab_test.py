__author__ = 'dcard'

import vocabulary
import vocabulary_with_counts


def main():
    vocab = vocabulary.Vocab('_i_')
    vocab.add_tokens(['a', 'a', 'b', 'c'])
    vocab.add_tokens(['a', 'd', 'e'])
    vocab.write_to_file('test.json')
    vocab.read_from_file('test.json')
    print vocab.index2token
    print vocab.token2index


    vocab2 = vocabulary_with_counts.VocabWithCounts('_i_')
    vocab2.add_tokens(['a', 'a', 'b', 'c'])
    vocab2.add_tokens(['a', 'd', 'e'])
    vocab2.write_to_file('test2.json')
    vocab2.read_from_file('test2.json')
    print vocab2.index2token
    print vocab2.token2index
    print vocab2.counts
    print vocab2.doc_counts


    print vocab2.get_counts_from_indices([0, 1, 2])



if __name__ == '__main__':
    main()

