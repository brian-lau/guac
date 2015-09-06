import sys
import operator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from ..util import file_handling as fh
from ..preprocessing import data_splitting as ds

# SVMNB and defaults taken from http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

# Wrapper class for (mostly) models from scikit-learn
class SparseModel:
    model = None
    model_type = None
    default = None
    column_names = None
    params = None
    trained = None
    w = None
    b = None

    def __init__(self, model_type=None, column_names=None, **kwargs):
        self.model_type = model_type
        self.column_names = column_names
        self.params = kwargs
        self.trained = None
        if model_type == 'LR':
            if self.params.get('regularization', None) is None:
                self.params['regularization'] = 'l1'
            if self.params.get('alpha', None) is None:
                self.params['alpha'] = 1.0
            self.model = lr(penalty=self.params['regularization'], C=self.params['alpha'])
        elif model_type == 'SVM' or model_type == 'SVMNB':
            if self.params.get('kernel', None) is None:
                self.params['kernel'] = 'rbf'

            if model_type == 'SVM':
                if self.params.get('alpha', None) is None:
                    self.params['alpha'] = 0.1
            else:  # elif model_type == SVMNB:
                self.params['kernel'] = 'linear'
                if self.params.get('alpha', None) is None:
                    self.params['alpha'] = 1
                if self.params.get('beta', None) is None:
                    self.params['beta'] = 0.25

            if self.params['kernel'] == 'linear':
                # override regularization parameter to avoid a conflict
                self.params['regularization'] = 'l2'
                self.model = svm.LinearSVC(C=self.params['alpha'])
            else:  # elif self.params['kernel'] != 'linear':
                if self.params.get('degree', None) is None:
                    self.params['degree'] = 3
                if self.params.get('gamma', None) is None:
                    self.params['gamma'] = 0.0
                if self.params.get('coef0', None) is None:
                    self.params['coef0'] = 0.0
                self.model = svm.SVC(C=self.params['alpha'], kernel=self.params['kernel'], degree=self.params['degree'],
                                     gamma=self.params['gamma'], coef0=self.params['coef0'])
        elif model_type == 'MNB':
            if 'alpha' not in self.params:
                self.params['alpha'] = 1.0
            self.model = MultinomialNB(alpha=self.params['alpha'], fit_prior=True)
        elif model_type == 'myMNB':
            if 'alpha' not in self.params:
                self.params['alpha'] = 1.0
            self.model = None
        else:
            self.model_type = 'default'
            self.model = None

    def set_alpha(self, alpha):
        self.params['alpha'] = alpha
        if self.model_type == 'LR':
            self.model.set_params(C=alpha)
        elif self.model_type == 'SVM' or self.model_type == 'SVMNB':
            self.model.set_params(C=alpha)
        elif self.model_type == 'MNB':
            self.model.set_params(alpha=float(alpha))

    def fit(self, X, y):
        n_items, n_features = X.shape
        if y.sum() == 0 or y.sum() == n_items or self.model_type == 'default':
            counts = np.bincount(y)
            y_mode = np.argmax(counts)
            self.default = y_mode
            self.model = None
            self.model_type = 'default'
        elif self.model_type == 'myMNB':
            # assumes y = {0,1}
            assert len(np.bincount(y)) == 2
            index_0 = np.array(y) == 0
            index_1 = np.array(y) == 1
            p = np.sum(X.toarray()[index_1, :], axis=0) + float(self.params['alpha'])
            q = np.sum(X.toarray()[index_0, :], axis=0) + float(self.params['alpha'])
            self.w = np.log((p / float(np.sum(np.abs(p)))) / (q / float(np.sum(np.abs(q)))))
            self.b = np.log(np.sum(index_1) / float(np.sum(index_0)))
            if np.isnan(self.w).any():
                sys.exit("nans encountered in " + str(self.__class__) + "." + self.fit.__name__)
        elif self.model_type == 'SVMNB':
            # assumes y = {0,1}
            assert len(np.bincount(y)) == 2
            # first calculate the inputs we would use for MNB
            X_dense = X.toarray()
            _, n_features = X_dense.shape
            index_0 = np.array(y) == 0
            index_1 = np.array(y) == 1
            # assume a hyperparameter for MNB of 1.0
            p = np.sum(X_dense[index_1, :], axis=0) + 1
            q = np.sum(X_dense[index_0, :], axis=0) + 1
            r = np.log((p / float(np.sum(np.abs(p)))) / (q / float(np.sum(np.abs(q)))))
            # multiply the feature counts by the log-count ratios from MNB (element-wise)
            X_dense = X_dense * r
            X_sparse = csr_matrix(X_dense)
            # then train the on the adjusted data
            self.model.fit(X_sparse, y)
            # finally, take a mixture of the two
            w = np.array(self.model.coef_)
            w_bar = np.sum(np.abs(w)) / float(n_features)
            self.model.coef_ = (1 - self.params['beta']) * w_bar + self.params['beta'] * w
        elif self.model_type == 'MNB':
            # this was giving me some errors, but it was only because I was using int8 matrices, which were
            # resulting in negative numbers in X
            if X.min() < 0:
                fh.pickle_data(X, 'X_err.npy')
                print self.model_type
                sys.exit("X is not non-negative!")
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        self.trained = True

    def tune_alpha(self, X, y, alpha_values, train_indices, valid_indices, reuser=None, verbose=1):
        train_f1s = []
        valid_f1s = []
        X_train = X[train_indices, :]
        X_valid = X[valid_indices, :]
        y_train = y[train_indices]
        y_valid = y[valid_indices]

        if verbose > 1:
            print "Train:", X_train.shape, " Dev:", X_valid.shape

        for alpha in alpha_values:
            self.set_alpha(alpha)
            self.fit(X_train, y_train)

            f1_train, acc_train = self.eval_f1_acc(X_train, y_train)
            f1_valid, acc_valid = self.eval_f1_acc(X_valid, y_valid)

            if reuser is not None:
                f1_valid = reuser.mask_value(f1_valid, f1_train)

            train_f1s.append(f1_train)
            valid_f1s.append(f1_valid)

        self.trained = False

        return train_f1s, valid_f1s

    # X = sparse matrix (indices x features)
    # y = vector (indices x 1)
    # alpha_values = list of alpha values to try
    # td_splits = vector ot train/dev split values (indices x 1)
    def tune_by_cv(self, X, y, alpha_values, td_splits, n_dev_folds, reuser=None, verbose=1):
        column_names = [str(l) for l in alpha_values]
        valid_f1_summary = pd.DataFrame(np.zeros([n_dev_folds, len(alpha_values)]),
                                        index=range(n_dev_folds), columns=column_names)

        for dev_fold in range(n_dev_folds):
            train_indices = np.array(td_splits != dev_fold)
            valid_indices = np.array(td_splits == dev_fold)

            train_f1s, valid_f1s = self.tune_alpha(X, y, alpha_values, train_indices, valid_indices, reuser=reuser,
                                                   verbose=verbose)

            valid_f1_summary.loc[dev_fold] = valid_f1s

            if verbose > 1:
                print dev_fold, valid_f1s

        mean_valid_f1s = valid_f1_summary.mean(axis=0)
        best_alpha = float(mean_valid_f1s.idxmax())
        self.set_alpha(best_alpha)
        self.trained = False
        return valid_f1_summary, best_alpha

    def get_coefs(self):
        if self.model_type == 'default' or self.model_type == 'SVM':
            return None
        elif self.model_type == 'myMNB':
            return zip(self.column_names, self.w)
        else:
            return zip(self.column_names, self.model.coef_[0])

    def predict(self, X):
        n, p = X.shape
        if self.model_type == 'default':
            predictions = self.default * np.ones(shape=[n, 1], dtype=int)
        elif self.model_type == 'myMNB':
            predictions = np.array((np.dot(X.toarray(), np.array(self.w)) + self.b) > 0, dtype=int)
        else:
            predictions = self.model.predict(X)
        return predictions

    def predict_log_probs(self, X):
        n, p = X.shape
        if self.model == 'LR' or self.model == 'MNB':
            log_probs = self.model.predict_log_proba(X)
        else:
            log_probs = np.zeros(shape=[n, 1])
        return log_probs

    def eval_f1_acc(self, X, y):
        predicted = self.predict(X)
        if np.isnan(predicted).any() or np.isnan(y).any():
            sys.exit("nans encountered " + str(self.__class__) + "." + self.eval_f1_acc.__name__)
        if y.sum() == 0 and predicted.sum() == 0:
            f1 = 1.0
        elif y.sum() == 0:
            f1 = 0.0
        elif predicted.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(y, predicted)
        acc = accuracy_score(y, predicted)
        return f1, acc

    def write_to_file(self, output_filename):
        if self.model_type == 'SVM':
            output = {'params': self.params, 'intercept': self.model.intercept_[0], 'model_type': self.model_type}
        elif self.model_type == 'LR':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.model.intercept_[0], 'coefs': coefs_sorted,
                      'model_type': self.model_type}
        elif self.model_type == 'MNB':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.model.intercept_[0], 'coefs': coefs_sorted,
                      'model_type': self.model_type}
        elif self.model_type == 'myMNB':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.b, 'coefs': coefs_sorted,
                      'model_type': self.model_type}
        else:
            output = {'default': self.default,  'model_type': self.model_type}
        fh.write_to_json(output, output_filename, sort_keys=False)


def baseline_model(y):
    model = SparseModel()
    return model