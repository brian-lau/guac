from core.experiment.experiment import run_group_experiment


run_group_experiment('test_dlrd', ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams,source=POS_tagged'],
                     model_type='LR', verbose=1)


