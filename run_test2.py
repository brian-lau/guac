from core.experiment import experiment
from core.experiment import experiment_conformal
from core.experiment import experiment_multilabel_test
from core.rnn import run_with_bayes_opt
from core.rnn import rnn_general
from core.rnn import rnn_minibatch


"""
experiment_conformal.run_group_experiment('test',
                     ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                     test_fold=0, feature_list=['ngrams'],
                     model_type='LR', verbose=1)
"""

result = rnn_minibatch.main()

print('name' + '\t' + str(-result['loss']) +
      '\t' + str(result['final_test_f1']) + '\nvalid_f1s:' + '\t' + str(result['valid_f1s']) + '\n')
print('test_f1s:' + '\t' + str(result['test_f1s']) + '\n')

#from core.rnn import rnn_general

#rnn_general.main()
