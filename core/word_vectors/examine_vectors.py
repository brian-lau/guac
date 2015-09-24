import gensim

from ..util import defines
from ..util import file_handling as fh

# Load word2vec vectors
print "Loading vectors"
vector_file = fh.make_filename(defines.vectors_dir, 'chars_word2vec_25', 'bin')
word2vec_vectors = gensim.models.Word2Vec.load(vector_file)

#for char in 'abcdefghijklmnopqrstuvwxyz'.upper():
#    print char
#    print word2vec_vectors.most_similar(positive=[char], topn=5)

print word2vec_vectors.most_similar(positive=['T'], topn=90)