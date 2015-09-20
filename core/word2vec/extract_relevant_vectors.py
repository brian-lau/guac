import pandas as pd

from ..util import defines
from ..util import file_handling as fh
import gensim


## THIS IS A GOOD IDEA, BUT DOES NOT WORK

input_filename = fh.make_filename(fh.makedirs(defines.resources_clusters_dir, 'input'), 'anes_text', 'txt')
lines = fh.read_text(input_filename)

print "Building token set"
token_set = set()
for line in lines:
    token_set.update(line.split())

all_tokens = list(token_set)
print len(token_set), "tokens"

# Load word2vec vectors
print "Loading vectors"
vector_file = defines.word2vec_vectors_filename
word2vec_vectors = gensim.models.Word2Vec.load_word2vec_format(vector_file, binary=True)

print "Loading anes vectors"
small_vocab_file = fh.make_filename(defines.vectors_dir, 'anes_word2vec_300', 'bin')
small_vectors = gensim.models.Word2Vec.load(small_vocab_file)

subset = {}
print "Overwriting vectors"
for t in all_tokens:
    if t in word2vec_vectors:
        subset[t] = word2vec_vectors[t]

print "Saving vectors"
output_filename = fh.make_filename(defines.vectors_dir, 'default_word2vec_300', 'json')
fh.write_to_json(subset, output_filename)
