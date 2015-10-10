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
