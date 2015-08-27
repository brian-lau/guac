import sys
import operator

import numpy as np
from scipy.sparse import csr_matrix

from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from ..util import file_handling as fh

# SVMNB and defaults taken from http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

# Wrapper class for (mostly) models from scikit-learn
class SparseModel:
    model = None
    model_type = None
    default = None
    column_names = None
    params = None
    w = None
    b = None

    def __init__(self, model_type=None, column_names=None, **kwargs):
        self.model_type = model_type
        self.column_names = column_names
        self.params = kwargs
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
            self.model = MultinomialNB(alpha=1/float(self.params['alpha']), fit_prior=True)
        elif model_type == 'myBinaryMNB':
            if 'alpha' not in self.params:
                self.params['alpha'] = 1.0
            self.model = None
        else:
            self.model_type = 'default'
            self.model = None

    def fit(self, X, y):
        n_items, n_features = X.shape
        if y.sum() == 0 or y.sum() == n_items or self.model_type == 'default':
            counts = np.bincount(y)
            y_mode = np.argmax(counts)
            self.default = y_mode
            self.model = None
            self.model_type = 'default'
        elif self.model_type == 'myBinaryMNB':
            # assumes y = {0,1}
            assert len(np.bincount(y)) == 2
            index_0 = np.array(y) == 0
            index_1 = np.array(y) == 1
            p = np.sum(X.toarray()[index_1, :], axis=0) + 1/float(self.params['alpha'])
            q = np.sum(X.toarray()[index_0, :], axis=0) + 1/float(self.params['alpha'])
            self.w = np.log((p / float(np.sum(np.abs(p)))) / (q / float(np.sum(np.abs(q)))))
            self.b = np.log(np.sum(index_1) / float(np.sum(index_0)))
            if np.isnan(self.w).any():
                sys.exit("nans encountered in " + str(self.__class__) + "." + self.fit.__name__)
        elif self.model_type == 'SVMNB':
            # assumes y = {0,1}
            assert len(np.bincount(y)) == 2
            X_dense = X.toarray()
            _, n_features = X_dense.shape
            index_0 = np.array(y) == 0
            index_1 = np.array(y) == 1
            p = np.sum(X_dense[index_1, :], axis=0) + 1/float(self.params['alpha'])
            q = np.sum(X_dense[index_0, :], axis=0) + 1/float(self.params['alpha'])
            r = np.log((p / float(np.sum(np.abs(p)))) / (q / float(np.sum(np.abs(q)))))
            X_dense = X_dense * r
            X_sparse = csr_matrix(X_dense)
            self.model.fit(X_sparse, y)
            w = np.array(self.model.coef_)
            w_bar = np.sum(np.abs(w)) / float(n_features)
            self.model.coef_ = (1 - self.params['beta']) * w_bar + self.params['beta'] * w
        else:
            self.model.fit(X, y)

    def get_coefs(self):
        if self.model_type == 'default' or self.model_type == 'SVM':
            return None
        elif self.model_type == 'myBinaryMNB':
            return zip(self.column_names, self.w)
        else:
            return zip(self.column_names, self.model.coef_[0])

    def predict(self, X):
        n, p = X.shape
        if self.model_type == 'default':
            predictions = self.default * np.ones(shape=[n, 1], dtype=int)
        elif self.model_type == 'myBinaryMNB':
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
            output = {'params': self.params, 'intercept': self.model.intercept_[0]}
        elif self.model_type == 'LR':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.model.intercept_[0], 'coefs': coefs_sorted}
        elif self.model_type == 'MNB':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.model.intercept_[0], 'coefs': coefs_sorted}
        elif self.model_type == 'myBinaryMNB':
            coefs_list = self.get_coefs()
            coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
            coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output = {'params': self.params, 'intercept': self.b, 'coefs': coefs_sorted}
        else:
            output = {'default': self.default}
        fh.write_to_json(output, output_filename, sort_keys=False)



def baseline_model(y):
    model = SparseModel()
    return model