import numpy as np
from scipy.stats import beta


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


## aici sa rulezi un k-means (il folosesti la initializarea clusterelor)

if __name__ == '__main__':

    weights, _ = stick_breaking_prior(1, 1, 50)
    print(f"weights = {weights}, sum(weights) = {sum(weights)}")
