import os
import re
import codecs
from subprocess import call

from ..util import defines
from ..util import file_handling as fh
from ..feature_extractors import normalizer


def write_all_raw_text_to_file(input_filename, output_dir):
    print "Separating text from index"
    data = fh.read_json(input_filename)
    text_filename = fh.make_filename(output_dir, 'text', 'txt')
    index_filename = fh.make_filename(output_dir, 'index', 'txt')

    keys = data.keys()
    keys.sort()

    texts = []
    indices = []

    for key in keys:
        text = data[key]
        texts.append(text)
        indices.append(key)

    fh.write_list_to_text(texts, text_filename)
    fh.write_list_to_text(indices, index_filename)

    return text_filename, index_filename


# Write each response to a separate file for parsing with CoreNLP
# (because each may contain multiple sentences)
def normalize_text(text_filename, index_filename, output_dir):
    print "Normalizing text"
    lines = fh.read_text(text_filename)
    index = fh.read_text(index_filename)
    filelist = []
    for line_index, key in enumerate(index):
        key = key.rstrip('\n')
        line = lines[line_index].rstrip('\n')
        normalized_filename = fh.make_filename(output_dir, key, 'txt')
        filelist.append(normalized_filename)
        line = normalizer.fix_basic_punctuation(line)
        with codecs.open(normalized_filename, 'w') as output_file:
            output_file.write(line)

    filelist_filename = fh.make_filename(output_dir, 'filelist', 'txt')
    fh.write_list_to_text(filelist, filelist_filename)
    return filelist_filename


def call_corenlp(filelist_filename, output_dir, annotators):
    stanford_dir = os.path.join(defines.base_dir, '..', '..', '..', '..', 'Tools', 'NLP', 'Stanford',
                                'stanford-corenlp-full-2013-11-12')
    os.chdir(stanford_dir)
    print "Starting corenlp pipeline"
    cmd = ['java', '-cp', '*', '-Xmx2g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
           '-annotators', annotators, '-filelist', filelist_filename, '-outputDirectory', output_dir]
    print ' '.join(cmd)
    call(cmd)

    filelist = fh.read_text(filelist_filename)
    xml_filelist = [fh.make_filename(output_dir, fh.get_basename(f.rstrip('\n')), 'txt.xml') for f in filelist]
    xml_filelist_filename = fh.make_filename(output_dir, 'xml_filelist', 'txt')
    fh.write_list_to_text(xml_filelist, xml_filelist_filename)
    return xml_filelist_filename


def parse_xml_files(xml_filelist_filename, output_dir):
    filelist = fh.read_text(xml_filelist_filename)
    parsed_files = {}
    for file in filelist:
        file = file.rstrip('\n')
        # peel off both .txt and .xml
        basename = fh.get_basename(fh.get_basename(file))
        parsed_file = parse_xml_output(file)
        parsed_files[basename] = parsed_file

    parsed_filename = fh.make_filename(output_dir, 'parsed', 'json')
    fh.write_to_json(parsed_files, parsed_filename, sort_keys=False)
    return parsed_filename


def parse_xml_output(xml_filename):
    raw_xml = fh.read_text(xml_filename)
    sentences = []
    line_index = 0
    while line_index < len(raw_xml):
        line = raw_xml[line_index].lstrip()
        match = re.search('<sentence id="(\d*)">', line)
        if match is not None:
            line_index, sentence = parse_sentence(raw_xml, line_index)
            sentences.append(sentence)
        line_index += 1

    return sentences


def parse_sentence(lines, start_line):
    sentence = []
    line_index = start_line + 1
    while line_index < len(lines):
        line = lines[line_index].lstrip()
        if re.search('</sentence>', line) is not None:
            return line_index, sentence
        match = re.search('<token id="(\d*)">', line)
        if match is not None:
            #print "Token", match.group(1)
            line_index, token = parse_token(lines, line_index)
            sentence.append(token)
        line_index += 1

def parse_token(lines, start_line):
    # TODO: Parse coreference and dependencies
    token = {}
    line_index = start_line + 1
    while line_index < len(lines):
        line = lines[line_index].lstrip()
        if re.search('</token>', line) is not None:
            return line_index, token
        match = re.search('<word>(.*)</word>', line)
        if match is not None:
            token['word'] = match.group(1)
        match = re.search('<lemma>(.*)</lemma>', line)
        if match is not None:
            token['lemma'] = match.group(1)
        match = re.search('<POS>(.*)</POS>', line)
        if match is not None:
            token['POS'] = match.group(1)
        match = re.search('<NER>(.*)</NER>', line)
        if match is not None:
            token['NER'] = match.group(1)
        match = re.search('<Speaker>(.*)</Speaker>', line)
        if match is not None:
            token['Speaker'] = match.group(1)
        line_index += 1


def write_tagged_text(parsed_filename, output_filename):
    data = fh.read_json(parsed_filename)

    tagged_text = {}
    for key, sentences in data.items():
        tagged_sentences = []
        for sentence in sentences:
            tagged_tokens = []
            for token in sentence:
                word = token.get('word', '__MISSING__')
                POS = token.get('POS', '__MISSING__')
                lemma = token.get('lemma', '__MISSING__')
                NER = token.get('NER', '__MISSING__')
                #tagged = word + '_' + POS
                tagged = POS + '_POS_'
                tagged_tokens.append(tagged)
            tagged_sentence = ' '.join(tagged_tokens)
            tagged_sentences.append(tagged_sentence)
        tagged_text[fh.get_basename(key)] = ' '.join(tagged_sentences)

    fh.write_to_json(tagged_text, output_filename, sort_keys=False)


def main():

    input_filename = defines.data_raw_text_file
    output_dir = fh.makedirs(defines.data_stanford_dir)
    """
    text_filename, index_filename = write_all_raw_text_to_file(input_filename, output_dir)
    normalized_dir = fh.makedirs(output_dir, 'normalized')
    filelist_filename = normalize_text(text_filename, index_filename, normalized_dir)
    annotators = 'tokenize,ssplit,pos,lemma,ner'
    #annotators = 'tokenize,ssplit,pos,lemma,ner,parse,dcoref,sentiment'
    xml_dir = fh.makedirs(output_dir, 'xml')
    xml_filelist_filename = call_corenlp(filelist_filename, xml_dir, annotators)
    parsed_filename = parse_xml_files(xml_filelist_filename, output_dir)
    """
    parsed_filename = fh.make_filename(output_dir, 'parsed', 'json')

    final_output_dir = defines.data_processed_text_dir
    output_filename = fh.make_filename(final_output_dir, 'POS_tags', 'json')
    write_tagged_text(parsed_filename, output_filename)


if __name__ == '__main__':
    main()
