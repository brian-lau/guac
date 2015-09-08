from core.experiment import experiment
from core.experiment import experiment_multilabel_test


experiment.run_group_experiment('test',
                     ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams'],
                     model_type='LR', verbose=1)


#from core.rnn import rnn_general

#rnn_general.main()
