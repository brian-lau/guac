import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats


from classifier_chain import ClassifierChain

class ECC:

    chains = None

    def __init__(self, model_type, codes, feature_names=None, alphas=None, n_chains=10, **kwargs):
        self.chains = []
        for i in range(n_chains):
            chain = ClassifierChain(model_type, codes, feature_names, alphas, **kwargs)
            self.chains.append(chain)

    def fit(self, orig_X, all_y):
        for i, chain in enumerate(self.chains):
            chain.fit(orig_X, all_y)

    def tune_by_cv(self, orig_X, all_y, alpha_values, td_splits, n_dev_folds, reuser=None, verbose=1):
        for i, chain in enumerate(self.chains):
            best_alphas = chain.tune_by_cv(orig_X, all_y, alpha_values, td_splits, n_dev_folds, reuser, verbose)

    def predict(self, orig_X, index, codes):
        predictions = []
        for i, chain in enumerate(self.chains):
            predictions.append(chain.predict(orig_X, index, codes))

        prediction_arrays = []
        for df in predictions:
            prediction_arrays.append(np.reshape(df.values, [len(index), len(codes), 1]))

        # take the most common prediction across the ensemble
        final_stack = np.concatenate(prediction_arrays, axis=2)
        final = stats.mode(final_stack, axis=2)[0][:, :, 0]

        final_df = pd.DataFrame(final, index=index, columns=codes)
        return final_df


    def save_models(self):
        pass