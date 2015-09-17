import re

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from ..util import file_handling as fh
from ..experiment import evaluation
from sparse_model import SparseModel
#from multilabel_model import MultilabelModel

class PowersetModel(SparseModel):

    model_type = None
    models = None
    codes = None
    powerset_labels = None
    powerset_index = None

    def __init__(self, model_type, codes, **kwargs):
        self.codes = codes
        code = ['powerset_label']
        SparseModel.__init__(self, model_type, code, **kwargs)

    def binary_vec_to_powerset_index(self, binary_vec):
        return self.powerset_index[str(binary_vec)]

    def powerset_index_to_binary_vec(self, powerset_index):
        binary_str = self.powerset_labels[powerset_index]
        return [int(i) for i in binary_str[1:-1].split(',')]

    def powerset_indices_to_matrices(self, powerset_indices):
        n_codes = len(self.powerset_index_to_binary_vec(powerset_indices[0]))
        all_y = np.zeros([len(powerset_indices), n_codes], dtype='int32')
        for i, y in enumerate(powerset_indices):
            all_y[i, :] = self.powerset_index_to_binary_vec(y)
        return all_y

    # all_y = np.array
    def binary_vectors_to_powerset_list(self, all_y):
        n, n_codes = all_y.shape
        self.powerset_labels = []
        self.powerset_index = {}
        powerset_y = []
        for i in range(n):
            key = str(all_y[i, :].tolist())
            index = self.powerset_index.get(key, None)
            if index is None:
                self.powerset_index[key] = len(self.powerset_labels)
                self.powerset_labels.append(key)

            powerset_y.append(self.powerset_index[key])
        return np.array(powerset_y)

    # X = sparse matrix
    # all_y = matrix or vector
    def fit(self, X, all_y):
        converted = False
        if len(all_y.shape) == 1:
            converted = True
        else:
            n, n_codes = all_y.shape
            if n_codes == 1:
                converted = True

        if converted:
            SparseModel.fit(self, X, all_y)
        else:
            powerset_y = self.binary_vectors_to_powerset_list(all_y)
            SparseModel.fit(self, X, powerset_y)


    # THIS Has kind of hacked together stuff from SparseModel and MultilabelModel...
    # all_y = DataFrame...
    def tune_by_cv(self, X, all_y, alpha_values, td_splits, n_dev_folds, reuser=None, verbose=1):
        y = self.binary_vectors_to_powerset_list(all_y.as_matrix())
        alphas = pd.DataFrame(np.zeros([1, 1]), index=['alpha'], columns=['powerset_label'])
        valid_f1_summary, best_alpha = SparseModel.tune_by_cv(self, X, y, alpha_values, td_splits, n_dev_folds,
                                                              reuser, verbose)
        alphas['alpha', 'powerset_label'] = best_alpha
        return alphas

    def predict(self, X, index=None, codes=None):
        n, p = X.shape
        powerset_predictions = SparseModel.predict(self, X)
        example_str = self.powerset_labels[0]
        example = [int(i) for i in example_str[1:-1].split(',')]
        n_codes = len(example)
        predictions = np.zeros([n, n_codes])
        for i in range(n):
            predictions[i, :] = self.powerset_index_to_binary_vec(powerset_predictions[i])

        if index is not None:
            predictions = pd.DataFrame(predictions, index=index, columns=codes)

        return predictions

    # THIS NEEDS IMPLEMENTIGN...
    # SHOUDL BE BALE TO GET IT FOM EVALUATOIUN IT HITNK
    def eval_f1_acc(self, X, powerset_y):
        predicted = self.predict(X)
        true = self.powerset_indices_to_matrices(powerset_y)
        n_items, n_tasks = true.shape
        y_vector = np.reshape(true, [n_items*n_tasks, 1])
        p_vector = np.reshape(predicted, [n_items*n_tasks, 1])

        f1 = f1_score(y_vector, p_vector)
        acc = accuracy_score(y_vector, p_vector)

        return f1, acc



    def calc_codewise_f1_acc(self, predictions, true):
        codes = true.columns
        f1s = pd.DataFrame(np.zeros(len(codes)), columns=codes)
        acc = pd.DataFrame(np.zeros(len(codes)), columns=codes)
        for code in codes:
            f1s.loc['f1', code], acc.loc['acc', code] = self.models[code].eval_f1_acc(self, predictions, true[code])
        return f1s, acc

    """
    def save_models(self, output_dir):
        codes = self.models.keys()
        for code in codes:
            output_filename = fh.make_filename(output_dir, re.sub(' ', '_', code), 'json')
            model = self.models[code]
            model.write_to_file(output_filename)
    """





