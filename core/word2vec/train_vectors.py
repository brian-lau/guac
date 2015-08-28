import os
import gensim
import logging

from ..util import defines

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


input_dir = os.path.join(defines.resources_clusters_dir, 'input')
sentences = MySentences(input_dir) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=1, size=300, hs=0, negative=5)

output = os.path.join(defines.vectors_dir, 'drld_reddit_word2vec_300.bin')
model.save(output)
