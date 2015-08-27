import re
import json
import codecs

from ..util import defines
from ..util import file_handling as fh

token_dir = defines.data_token_dir

input_filename = fh.make_filename(token_dir, 'ngrams_1_rnn', 'json')
rnn_ngrams = fh.read_json(input_filename)

groups_filename = fh.make_filename(defines.resources_group_dir, 'group_drld', 'txt')
groups = fh.read_text(groups_filename)

industry_occupation_filename = fh.make_filename(defines.resources_group_dir, 'industry_and_occupation', 'txt')
i_and_o_groups = fh.read_text(industry_occupation_filename)

output_filename = fh.make_filename(defines.resources_clusters_dir, 'drld_input', 'txt')

comments_filename = fh.make_filename(reddit_dir, 'comments', 'json')

with codecs.open(output_filename, 'w') as output_file:
    keys = rnn_ngrams.keys()
    for k in keys:
        question = k.split('_')[0]
        text = ' '.join(rnn_ngrams[k])
        # exclude the dem/rep dis/likes and industry/occupation text
        if question not in groups and question not in i_and_o_groups:
            output_file.write(text + '\n')

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

print "Read", success_count, "out of", total_count, "comments."


