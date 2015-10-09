from core.web import output_responses

from core.experiment import experiment
from core.experiment import experiment_conformal
from core.experiment import experiment_multilabel_test
from core.experiment import run_with_bayes_opt

#from core.rnn import run_with_bayes_opt

from core.rnn import rnn_general
from core.rnn import rnn_minibatch


"""
experiment_multilabel_test.run_group_experiment('best_LR_DRLD_ecc_test',
                    ['Democrat-Likes', 'Democrat-Dislikes', 'Republican-Likes', 'Republican-Dislikes'],
                    test_fold=0,
                    feature_list=['ngrams,binarize=False,min_doc_threshold=4',
                                  'ngrams,n=2,binarize=False,min_doc_threshold=1',
                                  'dataset',
                                  'ngrams,binarize=True,min_doc_threshold=5,source=decorated_likes',
                                  'ngrams,binarize=False,min_doc_threshold=1,source=decorated_dem',
                                  'brown,clusters=anes,binarize=True'],
                    model_type='LR', regularization='l1')
"""

#result = run_with_bayes_opt.main()


#from core.rnn import rnn_general

#rnn_general.main()

output_responses.main()
