import os
import re
import glob
import codecs
from optparse import OptionParser

import numpy as np

from hyperopt import fmin, tpe, hp, Trials, space_eval

import experiment
from ..util import defines
from ..util import file_handling as fh


output_filename = None
reuse = None

space = {
    'model': hp.choice('model', [
        {
            'model': 'SVM',
        },
        {
            'model': 'LR',
            'regularization': hp.choice('regularization', ['l1', 'l2'])
        },
        {
            'model': 'MNB'
        },
        {
            'model': 'SVMNB',
            'beta': hp.uniform('beta', 0, 1)
        } ]
    ),
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

        'like-dislike':
            hp.choice('like-dislike', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'ld_binarize': hp.choice('ld_binarize', ['True', 'False']),
                    'ld_min_doc_threshold': hp.choice('ld_min_doc_threshold', [1,2,3,4,5])
                }
            ] ),
        'dem-rep':
            hp.choice('dem-rep', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'dr_binarize': hp.choice('dr_binarize', ['True', 'False']),
                    'dr_min_doc_threshold': hp.choice('dr_min_doc_threshold', [1,2,3,4,5])
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
            ] ),
        'NER':
            hp.choice('NER', [
                {
                    'use': False
                },
                {
                    'use': True,
                }
            ] ),
        'brown_vectors':
            hp.choice('brown_vectors', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'bc_binarize': hp.choice('bc_binarize', ['True', 'False'])
                }
            ] )
    },
}


def call_experiment(args):
    kwargs = {}

    model = args['model']['model']
    if model == 'LR':
        kwargs['regularization'] = args['model']['regularization']
    elif model == 'SVMNB':
        kwargs['beta'] = args['model']['beta']
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
    if args['features']['like-dislike']['use']:
        base = 'ngrams' + \
               ',binarize=' + args['features']['like-dislike']['ld_binarize'] + \
               ',min_doc_threshold=' + str(args['features']['like-dislike']['ld_min_doc_threshold'])
        likes = base + ',source=decorated_likes'
        dislikes = base + ',source=decorated_dislikes'
        feature_list.append(likes)
        feature_list.append(dislikes)
    if args['features']['dem-rep']['use']:
        base = 'ngrams' + \
               ',binarize=' + args['features']['dem-rep']['dr_binarize'] + \
               ',min_doc_threshold=' + str(args['features']['dem-rep']['dr_min_doc_threshold'])
        dem = base + ',source=decorated_dem'
        rep = base + ',source=decorated_rep'
        feature_list.append(dem)
        feature_list.append(rep)
    if args['features']['POS_tags']['use']:
        pos_tags = 'ngrams' + \
                   ',binarize=' + args['features']['POS_tags']['pos_binarize'] + \
                   ',source=POS_tags'
        feature_list.append(pos_tags)
    if args['features']['NER']['use']:
        ner = 'ngrams,source=NER'
        feature_list.append(ner)
    if args['features']['brown_vectors']['use']:
        brown = 'brown,clusters=drld,binarize=' + args['features']['brown_vectors']['bc_binarize']
        feature_list.append(brown)

    datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']


    base_dir = fh.makedirs(defines.exp_dir, '_'.join(datasets), "test_fold_0")
    basename = fh.get_basename(output_filename)
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
    if reuse:
        result = experiment.run_group_experiment(name, datasets, 0, feature_list, model_type=model, reuse=True,
                                                 **kwargs)
    else:
        result = experiment.run_group_experiment(name, datasets, 0, feature_list, model_type=model, **kwargs)
    print result

    with codecs.open(output_filename, 'a') as output_file:
        output_file.write(name + '\t' + str(result['loss']) + '\t' + str(result['test_f1']) + '\n')

    return result



def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-o', dest='output_filename', default='bayes_opt',
                      help='Output filename')
    parser.add_option('--reuse', dest='reuse', action="store_true", default=False,
                      help='Use reusable holdout; default=%default')

    (options, args) = parser.parse_args()

    global output_filename, reuse
    output_filename = fh.make_filename(defines.exp_dir, options.output_filename, 'log')
    reuse = options.reuse

    with codecs.open(output_filename, 'w') as output_file:
        output_file.write(output_filename + '\n')
        output_file.write('reuse = ' + str(reuse) + '\n')

    trials = Trials()
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    print space_eval(space, best)
    print trials.losses()



if __name__ == '__main__':
    main()