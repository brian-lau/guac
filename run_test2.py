
#experiment.run_group_experiment(['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
#                     test_fold=0, feature_list=['ngrams,source=POS_tags'],
#                     model_type='LR', verbose=1, regularization='l2')


from core.rnn import rnn_general

rnn_general.main()
