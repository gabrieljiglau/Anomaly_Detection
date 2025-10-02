import numpy as np
from scipy.special import gammaln
from scipy.stats import beta
from sklearn.cluster import KMeans


def stick_breaking_prior(a: float, b: float, truncated_clusters: int):

    weights = np.zeros(truncated_clusters)
    beta_samples = beta.rvs(a, a/b, size=truncated_clusters)

    remaining_stick = 1
    for i, sample in enumerate(beta_samples):
        current_weight = sample * remaining_stick
        weights[i] = current_weight
        remaining_stick *= (1 - sample)  # remove a quantity proportional to the drawn sample

    weights[-1] += remaining_stick  # add what's left to the last cluster
    return weights, beta_samples


def _data_mean(x_train):

    data_mean = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        dimension_mean = x_train.iloc[:, i].mean()
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


def student_t_pdf(x_in, degrees_of_freedom, dim_data, cluster_mean, scale_matrix):

    scale_matrix = (scale_matrix * scale_matrix.transpose()) / 2

    nominator = gammaln((degrees_of_freedom + dim_data) / 2)
    denominator = gammaln(degrees_of_freedom / 2) * ((degrees_of_freedom * np.pi) ** dim_data / 2)
    # print(f"scale_matrix = {scale_matrix}")
    # print(f"np.linalg.det(scale_matrix) = {np.linalg.det(scale_matrix)}")
    denominator *= np.sqrt(np.linalg.det(scale_matrix))

    diff = (x_in - cluster_mean).reshape(-1, 1)
    free_term = (1 + (1 / degrees_of_freedom) * (diff.transpose() @ np.linalg.inv(scale_matrix) @ diff))
    # print(f"degrees_of_freedom = {degrees_of_freedom}")
    free_term **= ((degrees_of_freedom + dim_data) / -2)

    return (nominator / denominator) * free_term


def build_sample_covariance(x_train, no_clusters, soft_counts, weighted_means, responsibilities):
    covariance = []

    for k in range(no_clusters):
        cov_dim = 0
        for idx, row in enumerate(x_train):
            diff = np.array(row[1:]) - weighted_means[k]
            diff = diff.reshape(-1, 1)
            cov_dim += (diff @ diff.transpose()) * responsibilities[k, idx]

        # print(f"cov_dim = {cov_dim}")
        # print(f"soft_counts[k] = {soft_counts[k]}")  # should sum up to 1
        covariance.append(cov_dim / soft_counts[k])

    return np.array(covariance)


def build_coefficient(beta_0, miu_0, soft_count, weighted_mean):
    first_term = (beta_0 * soft_count) / (beta_0 + soft_count)
    diff = (weighted_mean - miu_0).reshape(-1, 1)

    return first_term * (diff @ diff.transpose())


if __name__ == '__main__':

    weights, _ = stick_breaking_prior(1, 1, 50)
    print(f"weights = {weights}, sum(weights) = {sum(weights)}")
