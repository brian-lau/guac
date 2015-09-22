import os
import re
import sys
import glob
import codecs
import datetime
from optparse import OptionParser

import numpy as np

from hyperopt import fmin, tpe, hp, Trials, space_eval

import rnn_general
import rnn_minibatch
from ..util import defines
from ..util import file_handling as fh


output_dirname = None
output_filename = None
reuse = None

space = {
    'init': {
        'initialize_vectors': hp.choice('initalize_vectors', [
            {'initialize_vectors': True},
            {'initialize_vectors': False, 'size': hp.quniform('size', 5, 100, 5) }
            ]
            ),
        'init_scale': hp.uniform('init_scale', 0, 1)
        },
    'arch': {
        'unit': hp.choice('unit', ['basic', 'GRU', 'LSTM']),
        'n_hidden': hp.quniform('n_hidden', 5, 100, 5),
        'window': hp.choice('window', [1, 3]),
        'add_DRLD': hp.choice('add_DRLD', [True, False]),
        'train_embeddings': hp.choice('train_embeddings', [True, False]),
        'pooling_method': hp.choice('pooling_method', ['max', 'attention1', 'attention2', 'last']),
        'bidirectional': hp.choice('bidirectional', [
            {'bidirectional': False},
            {'bidirectional': True, 'combine': hp.choice('combine', ['concat', 'max', 'mean'])}
        ])
    },
    'training': {
        'learning_rate': hp.loguniform('learning_rate', -6, -1),
        'lr_emb_fac': hp.uniform('lr_emb_fac', 0, 1),
        'decay_delay': hp.choice('decay_delay', [3, 4, 5, 6, 7, 8]),
        'decay_factor': hp.uniform('decay_factor', 0, 1),
        'OOV_noise': hp.choice('OOV_noise', [
            {'OOV_noise': False},
            {'OOV_noise': True, 'noise_prob': hp.loguniform('noise_prob', -6, -3)}
        ]
        ),
        'minibatch_size': hp.choice('minibatch_size', [1, 4, 16]),
        'classify_minibatch_size': hp.choice('classify_minibatch_size', [1, 64])
    }
    # 'regularization': {'dropout'... 'corruption'...}
}



def call_experiment(args):
    datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']
    basename = fh.get_basename(output_dirname)

    params = {}

    params['test_fold'] = 0
    params['min_doc_thresh'] = 1
    params['initialize_word_vectors'] = args['init']['initialize_vectors']['initialize_vectors']
    if params['initialize_word_vectors']:
        params['vectors'] = 'chars_word2vec_25'
    else:
        params['vectors'] = 'chars_word2vec' + str(args['init']['initialize_vectors']['size'])
    params['add_OOV_dim'] = False
    params['init_scale'] = args['init']['init_scale']
    params['win'] = args['arch']['window']
    params['train_embeddings'] = args['arch']['train_embeddings']
    params['add_DRLD'] = args['arch']['add_DRLD']
    params['rnn_type'] = args['arch']['unit']
    params['n_hidden'] = int(args['arch']['n_hidden'])
    params['pooling_method'] = args['arch']['pooling_method']
    if args['arch']['bidirectional']['bidirectional']:
        params['bidirectional'] = True
        params['bi_combine'] = args['arch']['bidirectional']['combine']
    else:
        params['bidirectional'] = False
        params['bi_combine'] = None
    params['lr'] = args['training']['learning_rate']
    params['lr_emb_fac'] = args['training']['lr_emb_fac']
    params['decay_delay'] = args['training']['decay_delay']
    params['decay_factor'] = args['training']['decay_factor']
    if args['training']['OOV_noise']['OOV_noise']:
        params['add_OOV_noise'] = True
        params['OOV_noise_prob'] = float(args['training']['OOV_noise']['noise_prob'])
    else:
        params['add_OOV_noise'] = False
        params['OOV_noise_prob'] = 0.0
    params['minibatch_size'] = int(args['training']['minibatch_size'])
    params['classify_minibatch_size'] = int(args['training']['classify_minibatch_size'])

    params['ensemble'] = False
    params['n_dev_folds'] = 1
    params['n_epochs'] = 100

    if reuse:
        params['reuse'] = True
        params['orig_T'] = 0.02
        params['tau'] = 0.005
    else:
        params['reuse'] = False

    params['seed'] = np.random.randint(0, 4294967294)
    params['verbose'] = 1
    params['save_model'] = False

    base_dir = fh.makedirs(defines.exp_dir, 'rnn')
    basename = fh.get_basename(output_dirname)
    existing_dirs = glob.glob(os.path.join(base_dir, basename + '*'))
    max_num = 0
    for dir in existing_dirs:
        match = re.search(basename + '_(\d+)', dir)
        if match is not None:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    name = fh.get_basename(output_filename) + '_' + str(max_num + 1)

    params['exp_name'] = name

    result = rnn_minibatch.main(params)

    with codecs.open(output_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + name + '\t' +
                          str(-result['loss']) + '\t' + str(result['final_test_f1']) + '\t' +
                          str(result['true_valid_f1s']) + '\t' + str(result['train_f1s']) +
                          '\t' + str(params['lr']) + '\n')

    print result
    return result



def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-o', dest='output_dirname', default='bayes_opt_rnn_chars',
                      help='Output directory name')
    parser.add_option('--reuse', dest='reuse', action="store_true", default=False,
                      help='Use reusable holdout; default=%default')

    (options, args) = parser.parse_args()

    global output_dirname, output_filename, reuse, search_alpha, space
    reuse = options.reuse
    output_dirname = options.output_dirname

    if reuse:
        output_dirname += '_reuse'

    output_filename = fh.make_filename(defines.exp_dir, fh.get_basename(output_dirname), 'log')

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(output_dirname + '\n')
        #output_file.write('reuse = ' + str(reuse) + '\n')

    trials = Trials()
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    print space_eval(space, best)
    print trials.losses()



if __name__ == '__main__':
    main()
