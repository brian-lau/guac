import ast
import numpy as np

from experiment.experiment_new import run_group_experiment

from hyperopt import fmin, tpe, hp, Trials, space_eval

from random import shuffle, sample

alphas = np.repeat(1, 33)

run_group_experiment('test_dlrd', ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams,binarize=True'],
                     min_alpha_exp=0, max_alpha_exp=0, model_type='SVMNB', best_alphas=alphas)


"""
                                        'anugrams,annotation=Dem,groups=[Democrat-Dislikes;Democrat-Likes]',
                                        'anugrams,annotation=Rep,groups=[Republican-Dislikes;Republican-Likes]',
                                        'anugrams,annotation=Like,groups=[Republican-Likes;Democrat-Likes]',
                                        'anugrams,annotation=Dislike,groups=[Republican-Dislikes;Democrat-Dislikes]'
                                        ])
"""


#experiment_new.main()