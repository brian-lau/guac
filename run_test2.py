from core.experiment import experiment
from core.experiment import experiment_conformal
from core.experiment import experiment_multilabel_test
from core.experiment import run_with_bayes_opt
#from core.rnn import run_with_bayes_opt
from core.rnn import rnn_general
from core.rnn import rnn_minibatch


"""
experiment_multilabel_test.run_group_experiment('powerset_test',
                     ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams'],
                     model_type='LR', verbose=1)
"""

result = run_with_bayes_opt.main()


#from core.rnn import rnn_general

#rnn_general.main()
