import sys
import re
import random

import numpy as np
import pandas as pd
from scipy import sparse

from ..util import file_handling as fh
from sparse_model import SparseModel
from multilabel_model import MultilabelModel

class ClassifierChain(MultilabelModel):

    order = None

    def __init__(self, model_type, codes, feature_names=None, alphas=None, **kwargs):

        # generate a random order over models
        self.order = list(codes)[:]
        random.shuffle(self.order)

        self.model_type = model_type
        self.models = {}

        # create models in the pre-determined order, adding one feature each time for the output of the previous model
        for i, code in enumerate(self.order):
            model = SparseModel(model_type, feature_names[:], **kwargs)
            if alphas is not None:
                model.set_alpha(alphas[i])
            self.models[code] = model
            feature_names.append(code)

    def fit(self, orig_X, all_y):
        X = orig_X.copy()
        for i, code in enumerate(self.order):
            y = all_y[code].as_matrix()
            model = self.models[code]
            model.fit(X, y)
            predictions = model.predict(X)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape((predictions.size, 1))
                predictions_sp = sparse.csc_matrix(predictions)
            elif predictions.shape[0] == 1:
                predictions_sp = sparse.csc_matrix(predictions.T)
            else:
                predictions_sp = sparse.csc_matrix(predictions)
            X = sparse.csr_matrix(sparse.hstack([X, predictions_sp]))

    def tune_by_cv(self, orig_X, all_y, alpha_values, td_splits, n_dev_folds, reuser=None, verbose=1):
        X = sparse.csc_matrix(orig_X)
        n, p = X.shape
        codes = all_y.columns
        n_codes = len(codes)
        alphas = pd.DataFrame(np.zeros([1, n_codes]), index=['alpha'], columns=codes)

        for i, code in enumerate(self.order):
            y = all_y[code].as_matrix()
            model = self.models[code]
            valid_f1_summary, best_alpha = model.tune_by_cv(X, y, alpha_values, td_splits, n_dev_folds,
                                                                        reuser=reuser, verbose=verbose)
            alphas.loc['alpha', code] = best_alpha
            predictions = model.predict(X)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape((predictions.size, 1))
                predictions_sp = sparse.csc_matrix(predictions)
            elif predictions.shape[0] == 1:
                predictions_sp = sparse.csc_matrix(predictions.T)
            else:
                predictions_sp = sparse.csc_matrix(predictions)

            X = sparse.csc_matrix(sparse.hstack([X, predictions_sp]))

            if verbose > 0:
                print i, code, y.sum(), best_alpha, valid_f1_summary.mean(axis=0)[str(best_alpha)]

        return alphas

    def predict(self, orig_X, index, codes):
        X = sparse.csc_matrix(orig_X)
        n, p = X.shape
        predictions_df = pd.DataFrame(np.zeros([n, len(codes)], dtype=int), index=index, columns=codes)

        for i, code in enumerate(self.order):
            model = self.models[code]
            predictions = model.predict(X)
            predictions_csc = sparse.csc_matrix(predictions)
            X = sparse.csr_matrix(sparse.hstack([X, predictions_csc.T]))
            predictions_df[code] = predictions

        return predictions_df

    def predict_log_probs(self, orig_X, index, codes):
        X = sparse.csc_matrix(orig_X)
        n, p = X.shape
        log_probs = pd.DataFrame(np.zeros([n, len(codes)], dtype=float), index=index, columns=codes)

        for i, code in enumerate(self.order):
            model = self.models[code]
            log_probs[code] = model.predict_p_y_eq_1(X)
            predictions = sparse.csc_matrix(model.predict(X))
            X = sparse.csr_matrix(sparse.hstack([X, predictions.T]))

        return log_probs

    def save_models(self):
        pass
