import re

def fix_basic_punctuation(text, lower=True, remove_angle_brackets=True, remove_double_quotes=True,
                          remove_backticks=True, split_on_dashes=True, replace_underscores=True,
                          sentence_split_on_slashes=True):
    if lower:
        text = text.lower()
    text = text.lstrip()
    text = text.rstrip()
    if remove_angle_brackets:
        text = re.sub('[<>]', '', text)
    if remove_double_quotes:
        text = re.sub('"', '', text)
    if remove_backticks:
        text = re.sub('`', '', text)
    if split_on_dashes:
        text = re.sub('-', ' - ', text)
    if replace_underscores:
        text = re.sub('_', ' - ', text)

    # replace the variable number of slashes used to separate sentences
    if sentence_split_on_slashes:
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

