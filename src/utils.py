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


def gaussian_pdf(instance, dim_data, covariance, mean):
    # instance is a real valued vector
    denominator =  (2 * np.pi) ** (dim_data / 2) * np.sqrt(np.linalg.det(covariance))
    diff = instance - mean
    cov_inverse = fractional_matrix_power(covariance, -1)
    exp_term = np.exp(-1/2 * (diff.transpose() @ cov_inverse @ diff))

    return exp_term / denominator


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
