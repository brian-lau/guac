from ..util import file_handling as fh

import html
import output_labels
import output_label_index
import output_responses
import output_response_index
import output_words
import output_word_index

def make_masthead(active_index):
    names = ['Responses', 'Labels', 'Words']
    targets = ['index_responses.html', 'index_labels.html', 'index_words.html']
    links = []
    for index, name in enumerate(names):
        target = targets[index]
        link = html.make_link(target, name)
        links.append(link)
    masthead = html.make_masthead(links, active_index)
    return masthead


def get_word_list(codes, model_dir):
    word_list = set()
    for code_index, code in enumerate(codes):
        # load coefficients from unigram model
        model_filename = fh.make_filename(model_dir, html.replace_chars(code), 'json')
        model = fh.read_json(model_filename)
        if 'coefs' in model:
            coefs = dict(model['coefs'])
            words = [word[4:] for word in coefs.keys()]
            word_list.update(words)
    word_list = list(word_list)
    word_list.sort()
    return word_list


def make_all_pages():
    output_labels.output_label_pages()
    output_label_index.output_label_index()
    output_responses.output_responses(dataset='Democrat-Likes')
    output_responses.output_responses(dataset='Democrat-Dislikes')
    output_responses.output_responses(dataset='Republican-Dislikes')
    output_responses.output_responses(dataset='Republican-Likes')
    output_response_index.output_response_index()
    output_words.output_words()
    output_word_index.output_word_index()


def main():
    make_all_pages()

if __name__ == '__main__':
    main()
