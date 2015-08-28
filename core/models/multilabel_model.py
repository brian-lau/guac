import pandas as pd

from .. preprocessing import data_splitting
from sparse_model import SparseModel

class MultilabelModel():

    model_type = None
    models = None

    def __init__(self, model_type, codes, column_names=None, reuse=True, **kwargs):
        self.model_type = model_type
        self.models = {}
        for code in codes:
            model = SparseModel(model_type, column_names, **kwargs)
            self.models[code] = model

    # X = sparse matrix
    # all_y = DataFrame of labels
    def fit(self, X, all_y):
        codes = all_y.columns
        for code in codes:
            y = all_y[code].as_matrix()
            model = self.models[code]
            model.fit(X, y)

    def tune_by_cv(self, X, all_y, alpha_values, td_splits, n_dev_folds, reuser=None, verbose=1):
        codes = all_y.columns
        for code in codes:
            y = all_y[code].as_matrix()
            self.models[code].tune_by_cv(X, y, alpha_values, td_splits, n_dev_folds, reuser, verbose)

    def predict(self):
        pass

    def eval_f1(self):
        pass


