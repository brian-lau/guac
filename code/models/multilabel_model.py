from .. preprocessing import data_splitting

from sparse_model import SparseModel

class MultilabelModel():

    model_type = None
    models = None

    def __init__(self, model_type=None, datasets=None, column_names=None, reuse=True, **kwargs):
        self.model_type = model_type

    def fit(self, X, y):
        pass

