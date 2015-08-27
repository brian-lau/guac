import os
import json
import codecs

import pandas as pd

from ..util import file_handling as fh, defines


def make_label_metaindex():
    input_filename = os.path.join('.', 'codes.json')
    with codecs.open(input_filename, 'r') as input_file:
        codes = json.load(input_file)

    label_index = {}

    for question in codes.keys():
        for mapping in codes[question]:
            orig = mapping[0]
            mine = int(mapping[1])
            if mine not in label_index.keys():
                label_index[mine] = {}
            label_index[mine][question] = orig

    return label_index


def get_labels(dataset):
    input_dir = defines.data_raw_labels_dir
    input_filename = fh.make_filename(input_dir, fh.get_basename(dataset), 'csv')
    label_data = pd.read_csv(input_filename, header=0, index_col=0)
    return label_data

