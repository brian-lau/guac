import re
import json
import codecs

from ..feature_extractors import tokenizer
from ..util import defines
from ..util import file_handling as fh

input_filename = defines.data_normalized_text_file
responses = fh.read_json(input_filename)

exclude_questions = ['Current-Industry', 'Current-Occupation', 'Past-Industry', 'Past-Occupation', 'drld-synthetic']

output_filename = fh.make_filename(defines.resources_clusters_dir, 'anes_text', 'txt')

#comments_filename = fh.make_filename(reddit_dir, 'comments', 'json')

#apos_words = {}

with codecs.open(output_filename, 'w') as output_file:
    keys = responses.keys()
    keys.sort()
    count = 0
    for k in keys:
        parts = k.split('_')
        if parts[0] not in exclude_questions:
            text = responses[k]
            tokens = []
            sentences = tokenizer.split_sentences(text)
            for s in sentences:
                sent_tokens = tokenizer.make_ngrams(s, 1, reattach=True, split_off_quotes=True)
                tokens = tokens + sent_tokens
                #for t in tokens:
                #    if re.search("'", t):
                #        apos_words[t] = 1
            output_file.write(' '.join(tokens) + '\n')
            count += 1
    """
    total_count = 0
    success_count = 0
    with codecs.open(comments_filename, 'r') as input_file:
        for line in input_file:
            total_count += 1
            if len(line) > 0:
                parts = line.split(': ')
                if parts[0] == 'data':
                    try:
                        data = json.loads(parts[1])
                        comment = data.get('body', '')
                        success_count += 1
                        paragraphs = comment.split('\n')
                        for p in paragraphs:
                            if re.search('[a-zA-Z]', p) is not None:
                                output_file.write(p + '\n')
                    except Exception as inst:
                        pass
    """


#print count
#keys = apos_words.keys()
#keys.sort()
#for k in keys:
#    print k