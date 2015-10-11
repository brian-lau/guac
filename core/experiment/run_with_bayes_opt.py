import os
import re
import sys
import glob
import codecs
import datetime
from optparse import OptionParser

import numpy as np

from hyperopt import fmin, tpe, hp, Trials, space_eval

import experiment
from ..util import defines
from ..util import file_handling as fh


output_dirname = None
output_filename = None
reuse = None
search_alpha = None
run = None
group = None
test_fold = None
n_dev_folds = None


space = {
    'features': {
        'unigrams':
            {
                'u_binarize': hp.choice('u_binarize', ['True', 'False']),
                'u_min_doc_threshold': hp.choice('u_min_doc_threshold', [1,2,3,4,5])
            },
        'bigrams':
            hp.choice('bigrams', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'b_binarize': hp.choice('b_binarize', ['True', 'False']),
                    'b_min_doc_threshold': hp.choice('b_min_doc_threshold', [1,2,3,4,5])
                }
            ] ),
        'dataset':
            hp.choice('dataset', [
                {
                    'use': False
                },
                {
                    'use': True,
                }
            ] ),
        'POS_tags':
            hp.choice('POS_tags', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'pos_binarize': hp.choice('pos_binarize', ['True', 'False']),
                }
            ]),
        'NER':
            hp.choice('NER', [
                {
                    'use': False
                },
                {
                    'use': True,
                }
            ]),
        'brown_vectors':
            hp.choice('brown_vectors', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'bc_binarize': hp.choice('bc_binarize', ['True', 'False'])
                }
            ])
    }
}

def add_drld():
    space['features']['like-dislike'] = hp.choice('like-dislike', [
            {
                'use': False
            },
            {
                'use': True,
                'ld_binarize': hp.choice('ld_binarize', ['True', 'False']),
                'ld_min_doc_threshold': hp.choice('ld_min_doc_threshold', [1, 2, 3, 4, 5])
            }
        ])
    space['features']['dem-rep'] = hp.choice('dem-rep', [
            {
                'use': False
            },
            {
                'use': True,
                'dr_binarize': hp.choice('dr_binarize', ['True', 'False']),
                'dr_min_doc_threshold': hp.choice('dr_min_doc_threshold', [1, 2, 3, 4, 5])
            }
        ])


def add_MIP():
    space['features']['personal'] = hp.choice('personal', [
        {
            'use': False
        },
        {
            'use': True,
            'per_binarize': hp.choice('per_binarize', ['True', 'False']),
            'per_min_doc_threshold': hp.choice('per_min_doc_threshold', [1, 2, 3, 4, 5])
        }
    ])


def add_MOLD():
    space['features']['like-dislike'] = hp.choice('like-dislike', [
        {
            'use': False
        },
        {
            'use': True,
            'ld_binarize': hp.choice('ld_binarize', ['True', 'False']),
            'ld_min_doc_threshold': hp.choice('ld_min_doc_threshold', [1, 2, 3, 4, 5])
        }
    ])
    add_obama()
    add_mccain()

def add_obama():
    space['features']['obama'] = hp.choice('obama', [
        {
            'use': False
        },
        {
            'use': True,
            'ob_binarize': hp.choice('ob_binarize', ['True', 'False']),
            'ob_min_doc_threshold': hp.choice('ob_min_doc_threshold', [1, 2, 3, 4, 5])
        }
    ])


def add_clinton():
    space['features']['clinton'] = hp.choice('clinton', [
        {
            'use': False
        },
        {
            'use': True,
            'oc_binarize': hp.choice('oc_binarize', ['True', 'False']),
            'oc_min_doc_threshold': hp.choice('oc_min_doc_threshold', [1,2,3,4,5])
        }
    ] )



def add_mccain():
    space['features']['mccain'] = hp.choice('mccain', [
        {
            'use': False
        },
        {
            'use': True,
            'om_binarize': hp.choice('om_binarize', ['True', 'False']),
            'om_min_doc_threshold': hp.choice('om_min_doc_threshold', [1,2,3,4,5])
        }
    ] )

def call_experiment(args):
    kwargs = {}




    model = args['model']['model']

    if model == 'LR':
        kwargs['regularization'] = args['model']['regularization']
    elif model == 'SVMNB':
        kwargs['beta'] = args['model']['beta']
    elif model == 'SVM':
        ktype = args['model']['kernel']['ktype']
        kwargs['kernel'] = ktype
        if ktype == 'poly':
            kwargs['degree'] = args['model']['kernel']['degree']
    feature_list = []
    unigrams = 'ngrams' + \
               ',binarize=' + args['features']['unigrams']['u_binarize'] + \
               ',min_doc_threshold=' + str(args['features']['unigrams']['u_min_doc_threshold'])
    feature_list.append(unigrams)
    if args['features']['bigrams']['use']:
        bigrams = 'ngrams,n=2' + \
                  ',binarize=' + args['features']['bigrams']['b_binarize'] + \
                  ',min_doc_threshold=' + str(args['features']['bigrams']['b_min_doc_threshold'])
        feature_list.append(bigrams)
    if args['features']['dataset']['use']:
        dataset = 'dataset'
        feature_list.append(dataset)
    if 'like-dislike' in args['features']:
        if args['features']['like-dislike']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['like-dislike']['ld_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['like-dislike']['ld_min_doc_threshold'])
            likes = base + ',source=decorated_likes'
            dislikes = base + ',source=decorated_dislikes'
            feature_list.append(likes)
            feature_list.append(dislikes)
    if 'dem-rep' in args['features']:
        if args['features']['dem-rep']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['dem-rep']['dr_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['dem-rep']['dr_min_doc_threshold'])
            dem = base + ',source=decorated_dem'
            rep = base + ',source=decorated_rep'
            feature_list.append(dem)
            feature_list.append(rep)
    if 'personal' in args['features']:
        if args['features']['personal']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['personal']['per_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['personal']['per_min_doc_threshold'])
            personal = base + ',source=decorated_personal'
            political = base + ',source=decorated_political'
            feature_list.append(personal)
            feature_list.append(political)
    if 'obama' in args['features']:
        if args['features']['obama']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['obama']['ob_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['obama']['ob_min_doc_threshold'])
            obama = base + ',source=decorated_obama'
            feature_list.append(obama)
    if 'clinton' in args['features']:
        if args['features']['clinton']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['clinton']['oc_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['clinton']['oc_min_doc_threshold'])
            clinton = base + ',source=decorated_clinton'
            feature_list.append(clinton)
    if 'mccain' in args['features']:
        if args['features']['mccain']['use']:
            base = 'ngrams' + \
                   ',binarize=' + args['features']['mccain']['om_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['mccain']['om_min_doc_threshold'])
            mccain = base + ',source=decorated_mccain'
            feature_list.append(mccain)


    if args['features']['POS_tags']['use']:
        pos_tags = 'ngrams' + \
                   ',binarize=' + args['features']['POS_tags']['pos_binarize'] + \
                   ',source=POS_tags'
        feature_list.append(pos_tags)
    if args['features']['NER']['use']:
        ner = 'ngrams,source=NER'
        feature_list.append(ner)
    if args['features']['brown_vectors']['use']:
        brown = 'brown,clusters=anes,binarize=' + args['features']['brown_vectors']['bc_binarize']
        feature_list.append(brown)

    if reuse:
        kwargs['reuse'] = True
    else:
        kwargs['reuse'] = False

    alphas = None
    if search_alpha:
        alphas = []
        for alpha in args['alphas']:
            alphas.append(float(alpha))
        kwargs['best_alphas'] = alphas

    base_dir = fh.makedirs(defines.exp_dir, '_'.join(group), "test_fold_" + str(test_fold))
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

    print feature_list
    result = experiment.run_group_experiment(name, group, test_fold, feature_list, model_type=model,
                                             n_dev_folds=n_dev_folds, **kwargs)
    print result

    with codecs.open(output_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + name + '\t' +
                          str(-result['loss']) + '\t' + str(result['test_f1']) + '\n')

    return result



def main():

    usage = "%prog <DRLD|MIP|MOLD|Primary|General|Terrorist|PK-Brown|PK-Roberts|PK-Pelosi|PK-Cheney>"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='model', default='LR',
                      help='Model: (LR|SVM|MNB|SVMNB); default=%default')
    parser.add_option('-t', dest='test_fold', default=0,
                      help='Test fold; default=%default')
    parser.add_option('-o', dest='output_dirname', default='bayes_opt',
                      help='Output directory name')
    parser.add_option('--reuse', dest='reuse', action="store_true", default=False,
                      help='Use reusable holdout; default=%default')
    parser.add_option('--alpha', dest='alpha', action="store_true", default=False,
                      help='Include alpha in search space (instead of grid search); default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds to use when tuning/evaluating; default=%default')

    #parser.add_option('--codes', dest='n_codes', default=33,
    #                  help='Number of codes (only matters with --alpha); default=%default')

    (options, args) = parser.parse_args()

    global output_dirname, output_filename, reuse, search_alpha, space, run, group, test_fold, n_dev_folds

    run = args[0]
    reuse = options.reuse
    search_alpha = options.alpha
    #n_codes = int(options.n_codes)
    output_dirname = options.output_dirname
    model = options.model
    test_fold = int(options.test_fold)
    n_dev_folds = int(options.n_dev_folds)

    # allow user to specfiy a particular choice of model
    if model == 'LR':
        space['model'] = {
            'model': 'LR',
            #'regularization': hp.choice('regularization', ['l1', 'l2'])
            'regularization': 'l1'
        }
    elif model == 'SVM':
        space['model'] = {
            'model': 'SVM',
            'kernel': hp.choice('ktype', [
                {'ktype': 'linear'},
                {'ktype': 'poly', 'degree': hp.choice('degree', [2, 3, 4])},
                {'ktype': 'rbf'}
            ]
            )
        }
    elif model == 'MNB':
        space['model'] = {
            'model': 'MNB'
        }
    elif model == 'SVMNB':
        space['model'] = {
            'model': 'SVMNB',
            'beta': hp.uniform('beta', 0, 1)
        }
    else:
        sys.exit('Choice of model not supported!')

    if run == 'DRLD':
        add_drld()
        group = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']
        n_codes = 33
    elif run == 'MIP':
        add_MIP()
        group = ['MIP-Personal-1', 'MIP-Personal-2', 'MIP-Political-1', 'MIP-Political-2']
        n_codes = 74
    elif run == 'MOLD':
        add_MOLD()
        group = ['McCain-Likes', 'McCain-Dislikes', 'Obama-Likes', 'Obama-Dislikes']
        n_codes = 34
    elif run == 'Primary':
        add_obama()
        add_clinton()
        group = ['Obama-Primary', 'Clinton-Primary']
        n_codes = 42
    elif run == 'General':
        add_obama()
        add_mccain()
        group = ['Obama-General', 'McCain-General']
        n_codes = 41
    elif run == 'Terrorists':
        group = [run]
        n_codes = 28
    elif run == 'PK-Brown':
        group = [run]
        n_codes = 14
    elif run == 'PK-Cheney':
        group = [run]
        n_codes = 12
    elif run == 'PK-Pelosi':
        group = [run]
        n_codes = 15
    elif run == 'PK-Roberts':
        group = [run]
        n_codes = 14
    else:
        sys.exit('Dataset not recognized')

    output_dirname += '_' + model

    if search_alpha:
        space['alphas'] = []
        for i in range(n_codes):
            space['alphas'].append(hp.loguniform('alpha' + str(i), -1.15, 9.2),)
        output_dirname += '_alphas'

    if reuse:
        output_dirname += '_reuse'
    else:
        output_dirname += '_noreuse'
    output_dirname += '_' + run

    if n_dev_folds != 5:
        output_dirname += '_' + str(n_dev_folds)

    output_filename = fh.make_filename(defines.exp_dir, fh.get_basename(output_dirname), 'log')

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(output_dirname + '\n')
        output_file.write('reuse = ' + str(reuse) + '\n')
        output_file.write('search alphas = ' + str(search_alpha) + '\n')

    trials = Trials()
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=40,
                trials=trials)

    print space_eval(space, best)
    print trials.losses()



if __name__ == '__main__':
    main()
