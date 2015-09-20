import os
import re
import gensim
import logging

from ..util import defines
from ..util import file_handling as fh

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if re.search('.txt$', fname) is not None:
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()
"""

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()



input_dir = os.path.join(defines.resources_clusters_dir, 'input')
input_filename = fh.make_filename(input_dir, 'reddit', 'txt')
sentences = MySentences(input_filename) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=1, size=100, hs=0, negative=10)

output = os.path.join(defines.vectors_dir, 'reddit_word2vec_100.bin')
model.save(output)
