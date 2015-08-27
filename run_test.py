from code.experiment import experiment_new

from hyperopt import fmin, tpe, hp, Trials, space_eval

space = {
    'model': hp.choice('model', [
        {
            'model': 'SVM',
            #'alpha': hp.loguniform('alpha', -4.6, 9.2)
        },
        {
            'model': 'LR',
            'regularizer': hp.choice('regularizer', ['l1', 'l2']),
            #'lambda': hp.loguniform('lambda', -4.6, 9.2)
        } ] ),

    'features': {
        'unigrams': {
            'u_binarize': hp.choice('u_binarize', ['True', 'False']),
            'u_min_doc_threshold': hp.choice('u_min_doc_threshold', [1,2,3,4]),
            'u_concat_oov_counts': hp.choice('u_concat_oov_counts', ['True', 'False'])
        },
        'bigrams': hp.choice('bigrams', [
            {
                'use': False,
            },
            {
                'use': True,
                'b_binarize': hp.choice('b_binarize', ['True', 'False']),
                'b_min_doc_threshold': hp.choice('b_min_doc_threshold', [1,2,3,4,5,6]),
                'b_concat_oov_counts': hp.choice('b_concat_oov_counts', ['True', 'False'])
            } ] )

        }
    }


def call_experiment(args):
    #print args
    model = args['model']['model']
    if model == 'LR':
        regularizer = args['model']['regularizer']
        #hyperparams = [args['model']['lambda'] for i in range(33)]
    else:
        regularizer = 'l1'
        #hyperparams = [args['model']['alpha'] for i in range(33)]
    feature_list = []
    unigrams = 'ngrams' + \
        ',binarize=' + args['features']['unigrams']['u_binarize'] + \
        ',min_doc_threshold=' + str(args['features']['unigrams']['u_min_doc_threshold']) + \
        ',concat_oov_counts=' + args['features']['unigrams']['u_concat_oov_counts']
    feature_list.append(unigrams)
    if args['features']['bigrams']['use']:
        bigrams = 'ngrams,n=2' + \
                   ',binarize=' + args['features']['bigrams']['b_binarize'] + \
                   ',min_doc_threshold=' + str(args['features']['bigrams']['b_min_doc_threshold']) + \
                   ',concat_oov_counts=' + args['features']['bigrams']['b_concat_oov_counts']
        feature_list.append(bigrams)

    #name = model + '_' + regularizer + '_' + '_'.join(feature_list) + '_' + str(hyperparams[0])
    name = model + '_' + regularizer + '_' + '_'.join(feature_list)
    datasets = ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes']
    print name
    return experiment_new.run_group_experiment(name, datasets, 0, feature_list,
                                               model_type=model, regularizer=regularizer)


trials = Trials()
best = fmin(call_experiment,
            space=space,
            algo=tpe.suggest,
            max_evals=3,
            trials=trials)

#rseed=np.random.randint(1, 4294967295)
#print best
print space_eval(space, best)
print trials.losses()
#for trial in trials.trials:
#    print trial



#run_group_experiment('profile_test', ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
#                                    0, ['ngrams'], model_type='SVM')



#experiment_new.main()