import pickle
import os.path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.exceptions import NotFittedError
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, psi, logsumexp
from scipy.stats import gamma, invwishart, multivariate_normal
from scipy.linalg import fractional_matrix_power
from dataclasses import dataclass
from src.utils import stick_breaking_prior, gaussian_pdf, student_t_pdf, map_clusters


def weights_expectations(sticks, truncated_clusters):

    """
    :param sticks: the weight for each cluster
    :param truncated_clusters: maximum clusters allowed
    :return: the total clusters which have the expectation(weight) > threshold (e.g. 1e-3)
    """

    expectations = np.zeros(truncated_clusters, dtype=float)
    for cluster in range(truncated_clusters):
        a_k = sticks[cluster].a_k
        b_k = sticks[cluster].b_k
        current_exp = a_k / (a_k + b_k)  # aici cica ar trebui np.log ??

        for j in range(cluster):
            a_j = sticks[j].a_k
            b_j = sticks[j].b_k
            current_exp *= b_j / (a_j + b_j)

        expectations[cluster] = current_exp

    expectations /= np.sum(expectations)
    # print(f"expectations = {expectations}")

    return expectations


def count_clusters(expectations):

    active_clusters = 0
    current_percentage = 0
    sorted_expectations = sorted(expectations, reverse=True)
    for weight in sorted_expectations:
        if current_percentage < 0.99:
            active_clusters += 1
        current_percentage += weight

    return active_clusters

def beta_expectations(sticks, truncated_clusters, eps=1e-9):

    total_exp = 0
    for cluster in range(truncated_clusters):
        # a, b are positive parameters
        a_k = max(sticks[cluster].a_k, eps)
        b_k = max(sticks[cluster].b_k, eps)
        total_exp += psi(b_k + eps) - psi(a_k + b_k + eps) # numerical stability

    print(f"total_exp = {total_exp}")
    return total_exp


def responsibilities_sum(current_k, responsibilities):

    """
    :param current_k: cluster index
    :param responsibilities: the responsibilities
    :return: Σ_i (Σ_j>k resp_ij)
    """

    if current_k >= responsibilities.shape[0] - 1:
        return 0.0

    return np.sum(responsibilities[current_k + 1: , :])


def weight_posterior(current_k, sticks):

    a_k = sticks[current_k].a_k
    b_k = sticks[current_k].b_k
    weight = a_k / (a_k + b_k)

    for j in range(current_k):
        a_j = sticks[j].a_k
        b_j = sticks[j].b_k
        weight *= b_j / (a_j + b_j)

    return weight


def log_likelihood_gaussian(x_train, dim_data, cluster_means, cov_matrices, mixing_weights):

    """
    :return: log (pi * gaussian(params)) for each instance
    """

    log_likelihood = 0
    for row in x_train:
        prob_sum = 0
        for i in range(len(cluster_means)):
            prob_sum += mixing_weights[i] * gaussian_pdf(row, dim_data, cov_matrices[i], cluster_means[i])
        log_likelihood += np.log(prob_sum) + 1e-12

    return log_likelihood


def log_likelihood_t(X, k_max, niw_posteriors, mixing_weights, toll=1e-300):

    """
    :return: log (pi * student_T(params)) for each instance
    """

    log_responsibilities = np.zeros((k_max, X.shape[0]))

    for k in range(k_max):
        for index, x_row in enumerate(X):

            if index % 10000 == 0:
                print(f"Now at instance {index}, cluster{ k}")

            nominator = (niw_posteriors[k].beta + 1) * niw_posteriors[k].p_lambda
            denominator = niw_posteriors[k].niu - len(x_row) + 1
            denominator *= niw_posteriors[k].beta
            scale_matrix = nominator / denominator

            log_pdf = student_t_pdf(x_row, niw_posteriors[k].niu, len(x_row), niw_posteriors[k].miu, scale_matrix)
            print(f"log pdf for instance {index} = {log_pdf}")
            log_responsibilities[k, index] = np.log(np.maximum(log_pdf, toll)) + np.log(mixing_weights[k])

    return np.exp(log_responsibilities)


def _data_mean(x_train):

    data_mean = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        dimension_mean = x_train[:, i].mean()
        data_mean[i] = dimension_mean
    return data_mean


def init_priors(no_clusters, x_train, dim_data, epsilon):

    """
    return priors: prior on weights, data_mean, strength_mean (the confidence in the data_mean),
    degree_of_freedom, scale_matrix
    """

    return [np.ones(no_clusters),
            _data_mean(x_train),
            1,
            x_train.shape[1] + 1,
            (np.eye(dim_data) * epsilon) + np.eye(dim_data)]


def init_clusters(k, x_train):

    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=1, random_state=13)
    kmeans.fit(x_train)

    return kmeans.cluster_centers_, kmeans.labels_


def build_sample_covariance(x_train, no_clusters, soft_counts, weighted_means, responsibilities):


    covariance = []
    for k in range(no_clusters):
        cov_dim = 0
        for idx, row in enumerate(x_train):
            diff = row - weighted_means[k]
            diff = diff.reshape(-1, 1)
            cov_dim += (diff @ diff.transpose()) * responsibilities[k, idx]

        # print(f"cov_dim = {cov_dim}")
        # print(f"soft_counts[k] = {soft_counts[k]}")  # should sum up to 1
        soft_counts[k] = max(soft_counts[k], 1e-12)
        covariance.append(cov_dim / soft_counts[k])
    return np.array(covariance)


def build_coefficient(beta_0, miu_0, soft_count, weighted_mean):
    first_term = (beta_0 * soft_count) / (beta_0 + soft_count)
    diff = (weighted_mean - miu_0).reshape(-1, 1)

    return first_term * (diff @ diff.transpose())


def sample_covariance(degrees_of_freedom, scale_matrix):
    return invwishart.rvs(df=degrees_of_freedom, scale=np.linalg.inv(scale_matrix))


def sample_mean(miu_point_estimate, covariance_matrix, strength_mean):
    covariance_matrix = np.array(covariance_matrix)
    covariance_matrix /= strength_mean
    return multivariate_normal(mean=miu_point_estimate, cov=covariance_matrix).rvs()


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

    def update_lambda(self, beta_0, miu_0, lambda_0, k, soft_counts, weighted_means, cov_matrix):

        observed_data = soft_counts[k] * cov_matrix[k]
        uncertainty_coefficient = build_coefficient(beta_0, miu_0, soft_counts[k], weighted_means[k, :])
        self.p_lambda = lambda_0 + observed_data + uncertainty_coefficient


class BayesianGaussianMixture:

    """
    Hierarchical Dirichlet Process (DP) for GMMs, using a truncated Stick-Breaking prior
    DP ~ G(alpha, h), where -alpha is the concentration parameter;
                    h -the base distribution (the prior for cluster parameters)
    """

    def __init__(self, alpha_posteriors, truncated_clusters=50):

        self.k = truncated_clusters  # up to k clusters maximum
        self.active_clusters = self.k
        self.priors = object.__new__(PriorHyperparameters)
        self.niw_posteriors = []
        self.sticks = [object.__new__(MixingWeight) for _ in range(self.k)]
        self.alpha_posteriors = np.array(alpha_posteriors)
        self.responsibilities = []


    def train(self, num_iterations: int, x_train: pd.DataFrame, y_train=None):

        # the assumption here is that param x_train is cleaned and standardized, and are stored as numpy arrays
        dim_data = x_train.shape[1]
        self.responsibilities = np.zeros((self.k, dim_data))

        a = float(self.alpha_posteriors[0])
        b = float(self.alpha_posteriors[1])

        cluster_means, labels = init_clusters(self.k, x_train)  # cheap KMeans for a decent initialization
        weights = stick_breaking_prior(a=1, b=1, truncated_clusters=self.k)

        h_priors = init_priors(no_clusters=self.k, x_train=x_train, dim_data=dim_data, epsilon=1e-6)
        self.priors = PriorHyperparameters(h_priors[1], h_priors[2], h_priors[3], h_priors[4], a, b)

        # initializations for prior and posteriors
        for i in range(self.k):
            self.niw_posteriors.append(NiwPosteriors(cluster_means[i], self.priors.beta_0, self.priors.niu_0, self.priors.lambda_0))
            self.sticks[i].a_k = a
            self.sticks[i].b_k = b
            self.sticks[i].weight = weights[i]

        log_likelihoods = []

        for iteration in range(num_iterations):
            print(f"Now at iteration {iteration}")

            self.variational_e_step(x_train, dim_data)
            self.variational_m_step(x_train)

            # only for testing with fixed k
            # acc = self._evaluate_performance(x_train, y_train, self.niw_posteriors, self.sticks)
            # print(f"Accuracy score = {acc}") 

            log_likelihood = log_likelihood_gaussian(x_train, dim_data,
                                                     [posterior.miu for posterior in self.niw_posteriors],
                                                     [posterior.p_lambda for posterior in self.niw_posteriors],
                                                     [stick.weight for stick in self.sticks])
            print(f"log_likelihood = {log_likelihood}")
            log_likelihoods.append(log_likelihood)


        # all log_likelihoods, alpha_posteriors, sticks and active clusters, since they will be used for plotting
        return self.niw_posteriors, self.alpha_posteriors, self.sticks,  self.active_clusters, log_likelihoods


    def predict(self, x_test, y_test, num_samples=30, posteriors='../models/posteriors.pkl'):

        # aici ar trebui sa actualizezi cum salvezi / incarci parametrii invatati ai algoritmului

        """
        :return: the sampled parameters from each gaussian(mean, cov_matrix), as long as the number of samples drawn
        """

        if os.path.exists(posteriors):
            with open(posteriors, 'rb') as f:
                saved_posteriors = pickle.load(f)
                self.niw_posteriors = saved_posteriors['niw_posteriors']
                self.alpha_posteriors = saved_posteriors['alpha_posteriors']
                self.sticks = saved_posteriors['sticks_list']

                sigma_samples = []
                mu_samples = []
                for cluster_idx in range (self.k):

                    sigma_sample = []
                    mu_sample = []
                    for sample in range(num_samples):
                        cov_matrix = sample_covariance(self.niw_posteriors[cluster_idx].beta,
                                                       np.linalg.inv(self.niw_posteriors[cluster_idx].p_lambda))
                        mu = sample_mean(self.niw_posteriors[cluster_idx].miu, cov_matrix, self.niw_posteriors[cluster_idx].niu)

                        sigma_sample.append(cov_matrix)
                        mu_sample.append(mu)

                    sigma_samples.append(sigma_sample)
                    mu_samples.append(mu_sample)

                sigma_samples = np.array(sigma_samples)
                mu_samples = np.array(mu_samples)

                acc = self._evaluate_performance(x_test, y_test, self.niw_posteriors, self.sticks)
                print(f"Accuracy on test data = {acc}")
                return mu_samples, sigma_samples, num_samples
        else:
            raise NotFittedError('The Bayesian Nonparametric GMM needs to be trained first !')

    def _evaluate_performance(self, x_np, y_np, niw_posteriors,sticks):
        
        y_np = y_np.flatten()
        cluster_assignments = []

        for idx, (x_in, y_true) in enumerate(zip(x_np, y_np)):
            result_probs = []
            result_instance = []
            for k in range(self.active_clusters):
                nominator = (niw_posteriors[k].beta + 1) * niw_posteriors[k].p_lambda
                denominator = niw_posteriors[k].niu - len(x_in) + 1
                denominator *= niw_posteriors[k].beta
                scale_matrix = nominator / denominator

                result_instance.append(student_t_pdf(x_in, niw_posteriors[k].niu, len(x_in), niw_posteriors[k].miu,
                                                     scale_matrix) * sticks[k].weight)
            result_probs.append(result_instance)
            # print(f"result_probs = {result_probs}")
            result_probs /= np.sum(result_probs)
            cluster_idx = np.argmax(result_probs)
            cluster_assignments.append(cluster_idx)

        # assigning the true labels to each gaussian component
        cluster_labels, mapping = map_clusters(y_np, cluster_assignments, self.active_clusters)
        acc = accuracy_score(y_np, cluster_labels)

        return acc

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

                log_pdf = student_t_pdf(np.array(x_row), self.niw_posteriors[cluster_idx].niu, dim_data,
                                        self.niw_posteriors[cluster_idx].miu, scale_matrix) + 1e-12 
                log_responsibilities[cluster_idx, idx] = log_pi_expectation + np.log(log_pdf)

        log_responsibilities -= logsumexp(log_responsibilities, axis=0, keepdims=True)
        self.responsibilities = np.exp(log_responsibilities)

    def variational_m_step(self, x_train):
        
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
            self.sticks[cluster].b_k = alpha_expectation + responsibilities_sum(cluster, self.responsibilities)

        # alpha parameters update
        pi_expectations = weights_expectations(self.sticks, self.k)
        self.active_clusters = count_clusters(pi_expectations)

        # don't change the active clusters, since the clusters with low weight are more likely anomalous
        print(f"active_clusters = {self.active_clusters}")

        self.alpha_posteriors[0] = self.priors.a + self.k
        self.alpha_posteriors[1] = self.priors.b - beta_expectations(self.sticks, self.k)

        print(f"alpha_a = {self.alpha_posteriors[0]}")
        print(f"alpha_b = {self.alpha_posteriors[1]}")

        for cluster in range (self.k):
            self.sticks[cluster].weight = pi_expectations[cluster]

        # new weighted mean
        weighted_means = np.zeros((self.k, x_train.shape[1]))
        for cluster in range(self.k):
            for idx, row in enumerate(x_train):
                weighted_means[cluster, :] += self.responsibilities[cluster, idx] * row

            if soft_counts[cluster] > 1e-12:
                weighted_means[cluster, :] /= soft_counts[cluster]
            else:
                weighted_means[cluster, :] = self.priors.miu_0

        sample_cov = build_sample_covariance(x_train, self.k, soft_counts, weighted_means, self.responsibilities)

        # NIW posterior parameter updates
        for cluster in range(self.k):

            self.niw_posteriors[cluster].update_beta(self.priors.beta_0, cluster, soft_counts)
            self.niw_posteriors[cluster].update_niu(self.priors.niu_0, cluster, soft_counts)
            self.niw_posteriors[cluster].update_miu(self.priors.beta_0, self.priors.miu_0,
                                                    cluster, soft_counts, weighted_means)
            self.niw_posteriors[cluster].update_lambda(self.priors.beta_0, self.priors.miu_0, self.priors.lambda_0,
                                                       cluster, soft_counts, weighted_means, sample_cov)


        return self.active_clusters, self.alpha_posteriors, self.sticks, self.niw_posteriors