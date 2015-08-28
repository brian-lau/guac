from core.experiment.experiment import run_group_experiment

run_group_experiment('test_dlrd', ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams,binarize=True'],
                     model_type='LR', verbose=1)


"""
                                        'anugrams,annotation=Dem,groups=[Democrat-Dislikes;Democrat-Likes]',
                                        'anugrams,annotation=Rep,groups=[Republican-Dislikes;Republican-Likes]',
                                        'anugrams,annotation=Like,groups=[Republican-Likes;Democrat-Likes]',
                                        'anugrams,annotation=Dislike,groups=[Republican-Dislikes;Democrat-Dislikes]'
                                        ])
"""


#experiment_new.main()