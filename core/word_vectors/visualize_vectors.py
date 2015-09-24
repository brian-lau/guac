import matplotlib.pyplot as plt
import gensim

import numpy as np
from sklearn.manifold import TSNE

from ..util import defines
from ..util import file_handling as fh

# Load word2vec vectors
print "Loading vectors"
vector_file = fh.make_filename(defines.vectors_dir, 'chars_word2vec_25', 'bin')
#vector_file = '/Users/dcard/Projects/CMU/ARK/guac/resources/vectors/chars_word2vec_25.bin'
#vector_file = '/Users/dcard/Projects/CMU/ARK/guac/resources/vectors/anes_word2vec_300.bin'
vectors = gensim.models.Word2Vec.load(vector_file)

index = vectors.index2word
n_vectors = len(index)
n_dims = len(vectors[index[0]])

print n_vectors, n_dims

X = np.zeros([n_vectors, n_dims])
for i, v in enumerate(index):
    X[i, :] = vectors[v]

model = TSNE(n_components=2, random_state=0)
result = model.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1], s=0, marker='.')
for i, v in enumerate(index):
    plt.text(result[i, 0], result[i, 1], v)
plt.show()

