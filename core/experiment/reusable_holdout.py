import numpy as np
from scipy import stats

class ReuseableHoldout:

    orig_T = None
    tau = None
    T_hat = None
    budget = None

    def __init__(self, T=0.04, tau=0.01, budget=None):
        self.orig_T = T
        self.tau = tau
        gamma = self.get_laplace_noise(4)
        self.T_hat = T + gamma
        if budget is not None:
            self.budget = budget

    def get_laplace_noise(self, sd):
        return stats.laplace.rvs(scale=sd*self.tau/np.sqrt(2))

    def mask_value(self, value, training_value):
        if self.budget is not None:
            if self.budget < 0:
                return None
        eta = self.get_laplace_noise(8)
        if np.abs(value - training_value) > (self.T_hat + eta):
            if self.budget is not None:
                self.budget -= 1
            xi = self.get_laplace_noise(2)
            gamma = self.get_laplace_noise(4)
            self.T_hat = self.orig_T + gamma
            return value + xi
        else:
            return training_value
