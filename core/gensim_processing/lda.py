from gensim import corpora, models, similarities

from pprint import pprint

from ..util import defines
from ..util import file_handling as fh

input_dir = fh.makedirs(defines.resources_clusters_dir, 'input')
input_filename = fh.make_filename(input_dir, 'anes_text', 'txt')

#lines = fh.read_text(input_filename)
#texts = [line.split() for line in lines]

dictionary = corpora.Dictionary(line.lower().split() for line in open(input_filename))
output_filename = fh.make_filename(defines.resources_gensim_dir, 'drld', 'dict')
dictionary.save(output_filename)

class MyCorpus(object):
    def __iter__(self):
        for line in open(input_filename):
            yield dictionary.doc2bow(line.split())

# memory friendly (only load one vector into RAM at a time
corpus = MyCorpus()
output_filename = fh.make_filename(defines.resources_gensim_dir, 'drld', 'mm')
corpora.MmCorpus.serialize(output_filename, corpus)

mm = corpora.MmCorpus(output_filename)
print(mm)

lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=50,
                               update_every=1, chunksize=100, passes=1)


for i in range(20):
    print(lda.print_topic(i))

