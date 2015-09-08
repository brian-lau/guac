from sklearn.datasets import fetch_20newsgroups

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from ..util import defines

def make_heads_up_comparison(subj0, subj1):

    text = {}

    cats = [subj0, subj1]
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)

    train_filenames = [fh.get_basename(f) for f in train['filenames']]
    test_filenames = [fh.get_basename(f) for f in test['filenames']]
    filenames = train_filenames + test_filenames

    labels = pd.DataFrame(np.zeros(len(filenames), dtype=int), index=filenames, columns=[subj1])

    labels[subj1] = train['target'] + test['target']

    for i, name in enumerate(train_filenames):
        text[name] = train['data'][i]

    for i, name in enumerate(test_filenames):
        text[name] = test['data'][i]

    output_dir = fh.makedirs(defines.base_dir, 'raw', 'labels')
    output_filename = fh.make_filename(output_dir, 'religion', 'json')
    labels.to_csv(output_filename)

    output_dir = fh.makedirs(defines.base_dir, 'raw', 'text')

