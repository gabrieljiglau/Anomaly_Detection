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
    priors: i) for niw data mean, belief of strength in the mean, degrees of freedom and the scale matrix
            ii) for alpha: a, b
    """

    miu_0: np.ndarray  # data mean
    beta_0: int  # belief of strength in the mean
    niu_0: int  # degrees of freedom
    lambda_0: np.ndarray  # scale matrix

    # alpha ~ Gamma(a_q, b_q)
    # (small alpha -few clusters with larger weights-, big alpha -more clusters with smaller weights-)
    a: float
    b: float


@dataclass
class MixingWeight:

    """
    each weight ~ Beta(a_k, b_k)
    """

    a_k: float
    b_k: float
    weight: float


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
    DP ~ G(alpha, h), where -alpha is the concentration parameter;
                    h - the base distribution (the prior for cluster parameters)
    """

    def __init__(self, truncated_clusters=50):

        self.k = truncated_clusters  # up to k clusters maximum
        self.active_clusters = self.k
        self.priors = object.__new__(PriorHyperparameters)
        self.niw_posteriors = [object.__new__(NiwPosteriors) for _ in range(self.k)]
        self.sticks = [object.__new__(MixingWeight) for _ in range(self.k)]
        self.alpha_posteriors = np.ones(2, dtype=int)
        self.responsibilities = []


    def train(self, num_iterations: int, x_train: pd.DataFrame, y_train=None, posteriors='../models/posteriors.pkl'):

        # the assumption here is that param x_train is cleaned and standardized

        x_train = x_train.to_numpy()
        dim_data = x_train.shape[1]
        self.responsibilities = np.zeros((self.k, dim_data))

        a = float(self.alpha_posteriors[0])
        b = float(self.alpha_posteriors[1])

        cluster_means, labels = init_clusters(self.k, x_train)  # cheap KMeans for a decent initialization
        weights = stick_breaking_prior(a=a, b=b, truncated_clusters=self.k)

        h_priors = init_priors(no_clusters=self.k, x_train=x_train, dim_data=dim_data, epsilon=1e-6)
        self.priors = PriorHyperparameters(h_priors[1], h_priors[2], h_priors[3], h_priors[4], a, b)

        # initializations for prior and posteriors
        for i in range(self.k):
            self.niw_posteriors[i].miu = cluster_means[i]
            self.niw_posteriors[i].beta = self.priors.beta_0
            self.niw_posteriors[i].niu = self.priors.niu_0
            self.niw_posteriors[i].p_lambda = self.priors.lambda_0
            self.sticks[i].a_k = a
            self.sticks[i].b_k = b
            self.sticks[i].weight = weights[i]

        for iteration in range(num_iterations):

            self.variational_e_step(x_train, dim_data)
            self.variational_m_step(x_train, dim_data)
            
        with open(posteriors, 'wb') as f:
            pickle.dump({'niw_posteriors': self.niw_posteriors,
                        'alpha_posteriors': self.alpha_posteriors,
                        'sticks': self.sticks,
                        'active_clusters': self.active_clusters}, f)

    def predict(self, x_test, y_test):
        # de mutat si functia de monitorizare a performantei aici
        pass
    
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

                nominator = (self.niw_posteriors[cluster_idx].beta + 1) * self.niw_posteriors[cluster_idx].p_lambda
                denominator = self.niw_posteriors[cluster_idx].niu - dim_data + 1
                denominator *= self.niw_posteriors[cluster_idx].beta
                scale_matrix = nominator / denominator

                log_pdf = student_t_pdf(np.array(x_row[1:]), self.niw_posteriors[cluster_idx].niu, dim_data,
                                        self.niw_posteriors[cluster_idx].miu, scale_matrix)
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

        # stick breaking parameters update
        alpha_expectation = self.alpha_posteriors[0] / self.alpha_posteriors[1]
        for cluster in range(self.k):
            self.sticks[cluster].a_k = 1 + soft_counts[cluster]
            self.sticks[cluster].b_k = alpha_expectation + responsibilities_sum(self.responsibilities, cluster)

        # alpha parameters update
        self.active_clusters = count_clusters(self.sticks, self.k)
        self.alpha_posteriors[0] = self.priors.a + self.active_clusters
        self.alpha_posteriors[1] = self.priors.b - beta_expectations(self.sticks, self.k)

        # new weighted mean
        weighted_means = np.zeros((self.k, x_train.shape[1]))
        for cluster in range(self.k):
            for idx, row in enumerate(x_train):
                weighted_means[cluster, :] += self.responsibilities[cluster, idx] * np.array(row[1:])
            weighted_means[cluster, :] /= soft_counts[cluster]

        # new weights for the sticks
        new_weights = stick_breaking_prior(float(self.alpha_posteriors[0]), float(self.alpha_posteriors[1]), self.k)
        for cluster in range (self.k):
            self.sticks[cluster].weight = new_weights[cluster]

        # NIW posterior parameter updates
        for cluster in range(self.k):

            self.niw_posteriors[cluster].update_beta(self.priors.beta_0, cluster, soft_counts)
            self.niw_posteriors[cluster].update_niu(self.priors.niu_0, cluster, soft_counts)
            self.niw_posteriors[cluster].update_miu(self.priors.beta_0, self.priors.miu_0, cluster, soft_counts, weighted_means)
            self.niw_posteriors[cluster].update_lambda(self.priors.beta_0, self.priors.miu_0, self.priors.lambda_0,
                                                 cluster, x_train, soft_counts, weighted_means, self.responsibilities)















