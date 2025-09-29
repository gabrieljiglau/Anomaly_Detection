import numpy as np
from scipy.stats import beta


def stick_breaking_prior(alpha: float, truncated_clusters: int):

    weights = np.zeros(truncated_clusters)
    beta_samples = beta.rvs(1, alpha, size=truncated_clusters)

    remaining_stick = 1
    for i, sample in enumerate(beta_samples):
        current_weight = sample * remaining_stick
        weights[i] = current_weight
        remaining_stick *= (1 - sample)  # remove a quantity proportional to what's left

    weights[-1] += remaining_stick  # add what's left to the last cluster, just in case
    return weights


if __name__ == '__main__':

    weights = stick_breaking_prior(1, 5)
    print(f"weights = {weights}, sum(weights) = {sum(weights)}")
