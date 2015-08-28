import gensim
import numpy as np
import pandas as pd

from ..util import defines
from ..util import file_handling as fh

def main():

    input_filename = fh.make_filename(defines.data_token_dir, 'ngrams_1_rnn_all', 'json')
    response_tokens = fh.read_json(input_filename)

    print "Building token set"
    token_set = set()
    for r in response_tokens:
        token_set.update(response_tokens[r])

    all_tokens = list(token_set)
    print len(token_set), "tokens"

    print "Loading brown clusters"
    brown_cluster_filename = fh.make_filename(defines.vectors_dir, 'brown_vectors', 'json')
    brown_clusters_data = fh.read_json(brown_cluster_filename)
    print brown_clusters_data.keys()
    brown_index = brown_clusters_data['index']
    brown_vectors = brown_clusters_data['vectors']
    brown_counts = brown_clusters_data['counts']

    print "Inverting brown cluster index"
    # invert the brown cluster index
    brown_clusters = {}
    for word in brown_index.keys():
        code = brown_index[word]
        if code in brown_clusters:
            brown_clusters[code].append(word)
        else:
            brown_clusters[code] = [word]

    # Load word2vec vectors
    print "Loading vectors"
    vector_file = defines.word2vec_vectors_filename
    word2vec_vectors = gensim.models.Word2Vec.load_word2vec_format(vector_file, binary=True)

    print "Computing weighted mean for unknown words"
    # get weighted average vector for each cluster, based on its elements that have word2vec vectors
    word2vec_dim = 300
    mean_vectors = {}
    for code in brown_clusters.keys():
        vector = np.zeros(word2vec_dim)
        count = 0
        total_weight = 0
        for word in brown_clusters[code]:
            if word in word2vec_vectors:
                weight = int(brown_counts[word])
                vector += word2vec_vectors[word] * weight
                count += 1
                total_weight += weight
        if count > 0:
            vector /= float(total_weight)
        else:
            print code, "has no representatives in word2vec"

        mean_vectors[code] = vector

    print "Creating dictionary of final vectors"
    final_vectors = pd.DataFrame(np.zeros([len(all_tokens), word2vec_dim]), index=all_tokens)
    for word in all_tokens:
        if word in word2vec_vectors:
            final_vectors.loc[word] = word2vec_vectors[word]
        elif word in brown_index:
            final_vectors.loc[word] = mean_vectors[brown_index[word]]

    print "Saving to file"
    output_filename = fh.make_filename(defines.vectors_dir, 'brown_augmented_word2vec_300', 'csv')
    final_vectors.to_csv(output_filename, header=False)



if __name__ == '__main__':
    main()
