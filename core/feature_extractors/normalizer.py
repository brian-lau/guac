import re

def fix_basic_punctuation(text):
    text = text.lower()
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub('[<>"`]', '', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('_', ' - ', text)

    # replace the variable number of slashes used to separate sentences
    text = re.sub('^/+[\s]*', '', text)
    text = re.sub('\.[\s]*/+[\s]*', '. ', text)
    text = re.sub('\?[\s]*/+[\s]*', '? ', text)
    text = re.sub('![\s]*/+[\s]*', '! ', text)
    text = re.sub('[\s,]*/+[\s]*', '. ', text)

    text = re.sub('^[\\\\]+[\s]*', '', text)
    text = re.sub('\.[\s]*[\\\\]+[\s]*', '. ', text)
    text = re.sub('\?[\s]*[\\\\]+[\s]*', '? ', text)
    text = re.sub('![\s]*[\\\\]+[\s]*', '! ', text)
    text = re.sub('[\s,]*[\\\\]+[\s]*', '. ', text)

    return text

