import os
import json
import codecs

import numpy as np


def main():
    nTrain = 1000
    nTest = 200
    nDev = 200

    nVocab = 6
    seqMin = 3
    seqMax = 8

    data = {}
    data['train_x'], data['train_y'] = generate_data(nTrain, nVocab, seqMin, seqMax)
    data['test_x'], data['test_y'] = generate_data(nTest, nVocab, seqMin, seqMax)
    data['dev_x'], data['dev_y'] = generate_data(nDev, nVocab, seqMin, seqMax)

    for i in range(10):
        print data['train_x'][i], data['train_y'][i]

    #output_dir = base_dir + 'data/rnn/'
    with codecs.open(os.path.join(output_dir, 'data.json'), 'w') as output_file:
        json.dump(data, output_file, indent=2)

def generate_data(n, nVocab, seqMin, seqMax):
    x = []
    y = []
    for i in range(n):
        seqLen = np.random.randint(seqMin, seqMax)
        seq = list(np.random.randint(0, nVocab, seqLen))
        x.append(seq)
        pairs = [[seq[i], seq[i+1]] for (i, v) in enumerate(seq[:-1])]
        if [1, 2] in pairs:
            y.append(1)
        elif [1, 3] in pairs:
            y.append(1)
        elif [1, 4] in pairs:
            y.append(1)
        elif [5, 2] in pairs:
            y.append(1)
        elif [5, 3] in pairs:
            y.append(1)
        elif [5, 4] in pairs:
            y.append(1)
        else:
            y.append(0)

    return x, y

if __name__ == '__main__':
    main()