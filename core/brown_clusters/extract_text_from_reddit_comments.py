import re
import sys
import json
import codecs
from ..feature_extractors import normalizer
from ..feature_extractors import tokenizer

input_filename = 'comments.json'
output_filename = 'comment_text.txt'

with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

body_count = 0
with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
    body_count = 0
    for line in lines:
        parts = re.split('^data: ', line)
        if len(parts) > 1:
            text = parts[1]
            try:
                data = json.loads(text, encoding='utf-8')
                if 'body' in data:
                    body = data['body']
                    body = body.encode('ascii', 'ignore')
                    body = re.sub('\n', ' ', body)
                    body = normalizer.fix_basic_punctuation(body)
                    tokens = []
                    sentences = tokenizer.split_sentences(body)
                    for s in sentences:
                        sent_tokens = tokenizer.make_ngrams(s, 1, reattach=True, split_off_quotes=True)
                        tokens = tokens + sent_tokens

                    output_file.write(' '.join(tokens) + "\n")
                    body_count += 1
            except:
                #e = sys.exc_info()[0]
                #print e
                pass


print("body count =", body_count)
print("total count =", len(lines))

