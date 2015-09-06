import re

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from sparse_model import SparseModel

class MultilabelModel():

    model_type = None
    models = None

    def __init__(self, model_type, codes, feature_names=None, alphas=None, **kwargs):
        self.model_type = model_type
        self.models = {}
        for i, code in enumerate(codes):
            model = SparseModel(model_type, feature_names, **kwargs)
            if alphas is not None:
                model.set_alpha(alphas[i])
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
        n_codes = len(codes)
        alphas = pd.DataFrame(np.zeros([1, n_codes]), index=['alpha'], columns=codes)
        for code in codes:
            y = all_y[code].as_matrix()

            valid_f1_summary, best_alpha = self.models[code].tune_by_cv(X, y, alpha_values, td_splits, n_dev_folds,
                                                                         reuser=reuser, verbose=verbose)
            alphas.loc['alpha', code] = best_alpha
            if verbose > 0:
                print code, y.sum(), best_alpha, valid_f1_summary.mean(axis=0)[str(best_alpha)]

        return alphas

    def set_alphas(self, alphas):
        for code in alphas.keys():
            self.models[code].set_alpha(alphas[code])

    def predict(self, X, index, codes):
        n, p = X.shape
        predictions = pd.DataFrame(np.zeros([n, len(codes)], dtype=int), index=index, columns=codes)
        for code in codes:
            predictions[code] = self.models[code].predict(X)
        return predictions

    def predict_log_probs(self, X, index, codes):
        n, p = X.shape
        log_probs = pd.DataFrame(np.zeros([n, len(codes)], dtype=float), index=index, columns=codes)
        for code in codes:
            log_probs[code] = self.models[code].predict_log_probs(X)
        return log_probs

    def calc_codewise_f1_acc(self, predictions, true):
        codes = true.columns
        f1s = pd.DataFrame(np.zeros(len(codes)), columns=codes)
        acc = pd.DataFrame(np.zeros(len(codes)), columns=codes)
        for code in codes:
            f1s.loc['f1', code], acc.loc['acc', code] = self.models[code].eval_f1_acc(self, predictions, true[code])
        return f1s, acc

    def save_models(self, output_dir):
        codes = self.models.keys()
        for code in codes:
            output_filename = fh.make_filename(output_dir, re.sub(' ', '_', code), 'json')
            model = self.models[code]
            model.write_to_file(output_filename)






