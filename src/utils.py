import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln
from scipy.stats import beta, invwishart, multivariate_normal
from scipy.linalg import fractional_matrix_power


def stick_breaking_prior(a: float, b: float, truncated_clusters: int):

    weights = np.zeros(truncated_clusters)
    beta_samples = beta.rvs(a, a/b, size=truncated_clusters)

    remaining_stick = 1
    for i, sample in enumerate(beta_samples):
        current_weight = sample * remaining_stick
        weights[i] = current_weight
        remaining_stick *= (1 - sample)  # remove a quantity proportional to the drawn sample

    weights[-1] += remaining_stick  # add what's left to the last cluster
    return weights


def student_t_pdf(x_in, degrees_of_freedom, dim_data, cluster_mean, scale_matrix):

    scale_matrix = (scale_matrix * scale_matrix.transpose()) / 2

    nominator = gammaln((degrees_of_freedom + dim_data) / 2)
    denominator = gammaln(degrees_of_freedom / 2) * ((degrees_of_freedom * np.pi) ** dim_data / 2)
    denominator *= np.sqrt(np.linalg.det(scale_matrix))

    diff = (x_in - cluster_mean).reshape(-1, 1)
    free_term = (1 + (1 / degrees_of_freedom) * (diff.transpose() @ np.linalg.inv(scale_matrix) @ diff))
    free_term **= ((degrees_of_freedom + dim_data) / -2)

    return float((nominator / denominator) * free_term)


def log_likelihood_t(x, k_max, niw_posteriors, mixing_weights, toll=1e-300):
    """
    :return: log (pi * student_T(params)) for each instance
    """

    log_responsibilities = np.zeros((k_max, x.shape[0]))

    for k in range(k_max):
        for index, x_row in enumerate(x):

            if index % 10000 == 0:
                print(f"Now at instance {index}, cluster{k}")

            nominator = (niw_posteriors[k].beta + 1) * niw_posteriors[k].p_lambda
            denominator = niw_posteriors[k].niu - len(x_row) + 1
            denominator *= niw_posteriors[k].beta
            scale_matrix = nominator / denominator

            log_pdf = student_t_pdf(x_row, niw_posteriors[k].niu, len(x_row), niw_posteriors[k].miu, scale_matrix)
            print(f"log pdf for instance at index: {index} = {log_pdf}")
            log_responsibilities[k, index] = np.log(np.maximum(log_pdf, toll)) + np.log(mixing_weights[k])

    return np.exp(log_responsibilities)


def gaussian_pdf(instance, dim_data, covariance, mean):
    # instance is a real valued vector
    denominator =  (2 * np.pi) ** (dim_data / 2) * np.sqrt(np.linalg.det(covariance))
    diff = instance - mean
    cov_inverse = fractional_matrix_power(covariance, -1)
    exp_term = np.exp(-1/2 * (diff.transpose() @ cov_inverse @ diff))

    return exp_term / denominator


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


def count_clusters(expectations):
    active_clusters = 0
    current_percentage = 0
    print(f"expectations = {expectations}")
    sorted_expectations = sorted(expectations, reverse=True)
    for weight in sorted_expectations:
        if current_percentage < 0.99:
            active_clusters += 1
        current_percentage += weight

    return active_clusters


def weight_posterior(sticks, current_k):

    a_k = sticks[current_k].a_k
    b_k = sticks[current_k].b_k
    weight = a_k / (a_k + b_k)

    for j in range(current_k):
        a_j = sticks[j].a_k
        b_j = sticks[j].b_k
        weight *= b_j / (a_j + b_j)

    return weight


def anomaly_statistics(detected_x, true_anomalies):

    detected_anomalies = 0
    for true_y in true_anomalies:
        if true_y in detected_x:
            detected_anomalies += 1

    return detected_anomalies


def map_clusters(true_labels, cluster_assignments, no_clusters):

    cm = confusion_matrix(true_labels, cluster_assignments, labels=range(no_clusters))
    # print(f"confusion_matrix = {cm}")
    rows, cols = linear_sum_assignment(-cm)

    mapping = {col: row for row, col in zip(rows, cols)}
    new_assignments = np.array([mapping[cluster] for cluster in cluster_assignments])

    return new_assignments, mapping

def non_active_clusters(x, niw_posteriors, k_max, mixing_weights):

    """
    old version, when I tried to get the clusters with the lowest number of instances based solely on the gaussian pdf,
    without the mixing weights; it performs poorly
    :return: the clusters with the lowest number of assigned instances
    """

    na_instances = []
    na_clusters = []
    cluster_instances = np.zeros(k_max)
    for idx, x_in in enumerate(x):

        if idx % 100000 == 0:
            print(f"Now at index {idx}")

        result_probs = []
        for k in range(k_max):
            nominator = (niw_posteriors[k].beta + 1) * niw_posteriors[k].p_lambda
            denominator = niw_posteriors[k].niu - len(x_in) + 1
            denominator *= niw_posteriors[k].beta
            scale_matrix = nominator / denominator

            #result_probs.append((student_t_pdf(x_in, niw_posteriors[k].niu, len(x_in), niw_posteriors[k].miu, scale_matrix))
                              # * mixing_weights[k])
            result_probs.append(gaussian_pdf(x_in, len(x_in), scale_matrix, niw_posteriors[k].miu) * mixing_weights[k])

        cluster_idx = np.argmax(result_probs)
        # print(f"result_probs = {result_probs}")
        # print(f"cluster_idx = {cluster_idx}")
        cluster_instances[cluster_idx] += 1

        """
        if cluster_idx > truncate_from:
            na_instances.append(idx)
            na_clusters.append(cluster_idx)
        """
    return na_instances, na_clusters, cluster_instances
