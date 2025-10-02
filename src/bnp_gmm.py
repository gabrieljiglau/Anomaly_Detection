import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from scipy.stats import gamma, beta
from scipy.special import psi, logsumexp
from utils import *

@dataclass
class PriorHyperparameters:

    """
    priors: data mean, belief of strength in the mean, degrees of freedom and the scale matrix
    """

    miu_0: np.ndarray  # data mean
    beta_0: int  # belief of strength in the mean
    niu_0: int  # degrees of freedom
    lambda_0: np.ndarray  # scale matrix


@dataclass
class MixingWeight:

    """
    each weight ~ Beta(a_k, b_k)
    """

    a_k: int
    b_k: int
    weight: float


class VariationalGamma:

    """
    prior for alpha (the concentration parameter) ~ Gamma(a, b)
    """

    def __init__(self, a: int=1, b: int=1):
        self.a = a
        self.b = b
        self.value = 1

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

    def update_beta(self, beta_0, k, soft_counts):
        self.beta = beta_0 + soft_counts[k]

    def update_niu(self, niu_0, k, soft_counts):
        self.niu = niu_0 + soft_counts[k]

    def update_miu(self, beta_0, miu_0, k, soft_counts, weighted_means):
        self.miu = beta_0 * miu_0 + soft_counts[k] * weighted_means[k, :]
        self.miu /= self.beta

    def update_lambda(self, beta_0, miu_0, lambda_0, k, x_train, soft_counts, weighted_means, responsibilities):

        covariance_matrix = build_sample_covariance(x_train, k, soft_counts, weighted_means, responsibilities)
        observed_data = soft_counts[k] * covariance_matrix
        uncertainty_coefficient = build_coefficient(beta_0, miu_0, soft_counts[k], weighted_means[k, :])

        self.p_lambda = lambda_0 + observed_data + uncertainty_coefficient


class BayesianNonparametricMixture:

    """
    approximation of the Dirichlet Process (DP) for GMMs, using a truncated Stick-Breaking prior
    """

    def __init__(self, alpha: VariationalGamma=None, truncated_clusters=50):
        """
        :param alpha: concentration parameter that has a prior, in order to be inferred from data
                      (small alpha -few clusters with larger weights-, big alpha -more clusters with smaller weights-)
        """

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = VariationalGamma()

        self.k = truncated_clusters  # up to k clusters maximum
        self.responsibilities = None
        self.priors = object.__new__(PriorHyperparameters) # the base distribution (the prior for cluster parameters)
        self.posteriors = [object.__new__(NiwPosteriors) for _ in range(self.k)]
        self.sticks = [object.__new__(MixingWeight) for _ in range(self.k)]


    def train(self, num_iterations: int, x_train: pd.DataFrame, y_train=None, a_k=1, b_k=1,
              posteriors='../models/posteriors.pkl'):

        # the assumption here is that param x_train is cleaned and standardized

        x_train = x_train.to_numpy()
        dim_data = x_train.shape[1]
        self.responsibilities = np.zeros((self.k, dim_data))

        cluster_means, labels = init_clusters(self.k, x_train)  # cheap KMeans for a decent initialization
        weights, beta_samples = stick_breaking_prior(a=1, b=1, truncated_clusters=self.k)

        h_priors = init_priors(no_clusters=self.k, x_train=x_train, dim_data=dim_data, epsilon=1e-6)
        self.priors = PriorHyperparameters(h_priors[1], h_priors[2], h_priors[3], h_priors[4])

        # initializations for prior and posteriors
        for i in range(self.k):
            self.posteriors[i].miu = cluster_means[i]
            self.posteriors[i].beta = self.priors.beta_0
            self.posteriors[i].niu = self.priors.niu_0
            self.posteriors[i].p_lambda = self.priors.lambda_0
            self.sticks[i].a_k = a_k
            self.sticks[i].b_k = b_k
            self.sticks[i].weight = weights[i]

        for iteration in range(num_iterations):

            self.variational_e_step(x_train, dim_data)
            self.variational_m_step(x_train, dim_data)
            
            
            
    
    def variational_e_step(self, x_train, dim_data):
        
        """
        update the responsibilities (assignment probabilities) for each cluster
        """

        log_responsibilities = np.zeros((self.k, x_train.shape[0]))
        for cluster_idx in range(self.k):

            expectation_component = 0
            for j in range(cluster_idx):
                a_j = self.sticks[j].a_k
                b_j = self.sticks[j].b_k
                expectation_component += psi(b_j) - psi(a_j + b_j)

            a_k = self.sticks[cluster_idx].a_k
            b_k = self.sticks[cluster_idx].b_k
            log_pi_expectation = psi(a_k) - psi(a_k + b_k) + expectation_component
            for idx, x_row in enumerate(x_train):

                nominator = (self.posteriors[cluster_idx].beta + 1) * self.posteriors[cluster_idx].p_lambda
                denominator = self.posteriors[cluster_idx].niu - dim_data + 1
                denominator *= self.posteriors[cluster_idx].beta
                scale_matrix = nominator / denominator

                log_pdf = student_t_pdf(np.array(x_row[1:]), self.posteriors[cluster_idx].niu, dim_data,
                                        self.posteriors[cluster_idx].miu, scale_matrix)
                log_responsibilities[cluster_idx, idx] = log_pi_expectation + np.log(log_pdf)

        log_responsibilities -= logsumexp(log_responsibilities, axis=0, keepdims=True)
        self.responsibilities = np.exp(log_responsibilities)

    def variational_m_step(self, x_train, dim_data):
        
        """
        update the posteriors for the concentration parameter, niw and the sticks
        """

        soft_counts = np.zeros(self.k)
        for k in range(self.k):
            soft_counts[k] = np.sum(self.responsibilities[k, :])

        # stick breaking weights update
        for k in range(self.k):
            self.alpha.a = 1 + soft_counts[k]
            self.alpha.b = E[q(alpha)] + sum(i) sum(j > k) resp(i, j)

        # new weighted mean
        weighted_means = np.zeros((self.k, x_train.shape[1]))
        for k in range(self.k):
            for idx, row in enumerate(x_train):
                weighted_means[k, :] += self.responsibilities[k, idx] * np.array(row[1:])
            weighted_means[k, :] /= soft_counts[k]

        # NIW posterior parameter updates
        for k in range(self.k):

            self.posteriors[k].update_beta(self.priors.beta_0, k, soft_counts)
            self.posteriors[k].update_niu(self.priors.niu_0, k, soft_counts)
            self.posteriors[k].update_miu(self.priors.beta_0, self.priors.miu_0, k, soft_counts, weighted_means)
            self.posteriors[k].update_lambda(self.priors.beta_0, self.priors.miu_0, self.priors.lambda_0,
                                             k, x_train, soft_counts, weighted_means, self.responsibilities)















