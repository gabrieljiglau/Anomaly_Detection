import numpy as np
from pandas import read_feather
from dataclasses import dataclass

@dataclass
class PriorHyperparameters:

    """
    priors for the Normal Inverse Wishart distribution(data mean, belief of strength in the mean, degrees of freedom
                                                       and the scale matrix)
    """

    miu_0: np.ndarray  # data mean
    beta_0: int  # belief of strength in the mean
    niu_0: int  # degrees of freedom
    lambda_0: np.ndarray  # scale matrix

class PosteriorParameters:

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


class DirichletProcess:

    """
    approximation of the Dirichlet Process (DP) for GMMs, using a truncated Stick-Breaking prior
    """

    def __init__(self, alpha: float, h: PriorHyperparameters):
        """
        :param alpha: concentration parameter (small alpha -few clusters with larger weights-,
                                               big alpha -more clusters with smaller weights-)
        :param h: the base distribution (the prior for cluster parameters)
        """
        self.alpha = alpha
        self.h = h
















