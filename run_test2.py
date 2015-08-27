from code.experiment.experiment import run_group_experiment

run_group_experiment('test_dlrd', ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams,binarize=True'],
                     min_alpha_exp=0, max_alpha_exp=0, model_type='SVMNB')


"""
                                        'anugrams,annotation=Dem,groups=[Democrat-Dislikes;Democrat-Likes]',
                                        'anugrams,annotation=Rep,groups=[Republican-Dislikes;Republican-Likes]',
                                        'anugrams,annotation=Like,groups=[Republican-Likes;Democrat-Likes]',
                                        'anugrams,annotation=Dislike,groups=[Republican-Dislikes;Democrat-Dislikes]'
                                        ])
"""


#experiment_new.main()