import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from scipy.stats import gamma, beta
from scipy.special import psi
from utils import *

@dataclass
class NiwPriors:

    """
    priors for the Normal Inverse Wishart distribution(data mean, belief of strength in the mean, degrees of freedom
                                                       and the scale matrix)
    """
    miu_0: np.ndarray  # data mean
    beta_0: int  # belief of strength in the mean
    niu_0: int  # degrees of freedom
    lambda_0: np.ndarray  # scale matrix


@dataclass
class StickPriors:
    a_0: int
    b_0: int


class GammaDistribution:

    def __init__(self, a: int=1, b: int=1):
        self.a = a
        self.b = b
        self.p_alpha = 1

    def update_posterior(self, k, variational_a, variational_b):

        new_a = self.a + k - 1
        new_b = self.b

        for i in range(1, k - 1):
            new_b -= psi(variational_b[i]) - psi(variational_a[i] + variational_b[i])

        return new_a, new_b


class NiwPosteriors:

    """
    posteriors for the NIW distribution
    """

    def __init__(self, miu, beta, niu, p_lambda):
        self.miu = miu
        self.beta = beta
        self.niu = niu
        self.p_lambda = p_lambda

    def update_beta(self, beta_0, soft_counts):
        self.beta = beta_0 + soft_counts

    def update_niu(self, niu_0, soft_counts):
        self.niu = niu_0 + soft_counts

    def update_miu(self, beta_0, miu_0, k, soft_counts, weighted_means):
        self.miu = beta_0 * miu_0 + soft_counts[k] * weighted_means[k, :]
        self.miu /= self.beta

    def update_lambda(self, lambda_0, observed_data, uncertainty_coefficient):
        self.p_lambda = lambda_0 + observed_data + uncertainty_coefficient


class BayesianNonparametricMixture:

    """
    approximation of the Dirichlet Process (DP) for GMMs, using a truncated Stick-Breaking prior
    """

    def __init__(self, alpha: GammaDistribution, h: NiwPriors, truncated_clusters=50):
        """
        :param alpha: concentration parameter that has a prior, in order to be inferred from data
                      (small alpha -few clusters with larger weights-, big alpha -more clusters with smaller weights-)
        :param h: the base distribution (the prior for cluster parameters)
        """

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = GammaDistribution()

        self.h = h
        self.k = truncated_clusters
        self.responsibilities = None
        self.a_sticks = [object.__new__(StickPriors) for _ in range(self.k)]

    def train(self, num_iterations: int, x_train: pd.DataFrame, y_train=None, posteriors='../models/posteriors.pkl'):

        dim_data = x_train.shape[1]
        self.responsibilities = np.zeros((k, dim_data))















