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
from ..util import defines
from ..util import file_handling as fh


output_dirname = None
output_filename = None
reuse = None

space = {
    'input': { 'min_doc_thresh': hp.choice('min_doc_thresh', [1, 2, 3, 4]) },
    'init': {
        'vectors': hp.choice('vectors', [
            {'vectors': 'word2vec'},
            #{'vectors': 'anes', 'size': hp.choice('size', [100, 200, 300])}
            {'vectors': 'anes'}
        ]
        ),
        'add_OOV': hp.choice('add_OOV', [True, False]),
        'init_scale': hp.uniform('init_scale', 0, 1)

        },
    'arch': {
        'window': hp.choice('window', [1, 3]),
        'add_DRLD': hp.choice('add_DRLD', [True, False]),
        'pooling_method': hp.choice('pooling_method', ['max', 'mean', 'attention1', 'attention2']),
        'bidirectional': hp.choice('bidirectional', [
            {'bidirectional': False},
            {'bidirectional': True, 'combine': hp.choice('combine', ['concat', 'max', 'mean'])}
        ])
    },
    'training': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'lr_emb_fac': hp.uniform('lr_emb_fac', 0, 1),
        'decay_delay': hp.choice('decay_delay', [3, 4, 5, 6, 7, 8]),
        'decay_factor': hp.uniform('decay_factor', 0, 1),
        'OOV_noise': hp.choice('OOV_noise', [
            {'OOV_noise': False},
            {'OOV_noise': True, 'noise_prob': hp.loguniform('noise_prob', -6, -3)}
        ]
        ),
    }
    # 'regularization': {'dropout'... 'add_noise'..., 'corruption'...}
}



def call_experiment(args):
    datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']
    basename = fh.get_basename(output_dirname)

    params = {}

    params['test_fold'] = 0
    params['min_doc_thresh'] = args['input']['min_doc_thresh']
    params['initialize_word_vectors'] = True
    if args['init']['vectors'] == 'word2vec':
        params['vectors'] = 'default_word2vec'
        params['word2vec_dim'] = 300
    else:
        params['vectors'] = 'anes_word2vec'
        params['word2vec_dim'] = 300
    params['add_OOV'] = args['init']['add_OOV']
    params['init_scale'] = args['init']['init_scale']
    params['win'] = args['arch']['window']
    params['add_DRLD'] = args['arch']['add_DRLD']
    params['rnn_type'] = args['arch']['unit']
    params['n_hidden'] = args['arch']['n_hidden']
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
        params['OOV_noise_prob'] = args['training']['OOV_noise']['noise_prob']
    else:
        params['add_OOV_noise'] = False
    params['OOV_noise_prob'] = args['training']

    params['n_dev_folds'] = 5
    params['n_epochs'] = 40

    if reuse:
        params['reuse'] = True
        params['orig_T'] = 0.04
        params['tau'] = 0.01
    else:
        params['reuse'] = False

    params['seed'] = np.random.randint(0, 4294967294)
    params['verbose'] = 1
    params['save_model'] = True

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

    result = rnn_general.main(params)

    with codecs.open(output_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + name + '\t' +
                          str(-result['loss']) + '\t' + str((result['median_test_f1'])) + '\n')
        output_file.write('valid_f1s:' + '\t' + str(result['valid_f1s']) + '\n')
        output_file.write('test_f1s:' + '\t' + str(result['test_f1s']) + '\n')

    print result
    return result



def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='model', default='basic',
                      help='Model: (basic|GRU|LSTM); default=%default')
    parser.add_option('-o', dest='output_dirname', default='bayes_opt_rnn',
                      help='Output directory name')
    parser.add_option('--reuse', dest='reuse', action="store_true", default=False,
                      help='Use reusable holdout; default=%default')

    (options, args) = parser.parse_args()

    global output_dirname, output_filename, reuse, search_alpha, space
    reuse = options.reuse
    output_dirname = options.output_dirname
    model = options.model

    if model == 'basic':
        space['arch']['unit'] = 'basic'
        space['arch']['n_hidden'] = hp.quniform('n_hidden', 5, 300, 5)
    elif model == 'GRU':
        space['arch']['unit'] = 'GRU'
        space['arch']['n_hidden'] = hp.quniform('n_hidden', 5, 100, 5)
    elif model == 'LSTM':
        space['arch']['unit'] = 'LSTM'
        space['arch']['n_hidden'] = hp.quniform('n_hidden', 5, 75, 5)
    else:
        sys.exit('Model not supported!')

    output_dirname += '_' + model

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
