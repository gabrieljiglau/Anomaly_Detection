import os
import pickle
import numpy as np
from numpy import shape
from pandas import read_feather
from bnp_gmm import BayesianNonparametricMixture
from utils import (weight_posterior, instance_log_likelihood, anomaly_statistics, non_active_instances,
                   na_cluster_probs, na_mixing_weights)


if __name__ == '__main__':

    df = read_feather('../datasets/processed/creditcard_standardized.feather')
    X = df.iloc[:, :-1]
    X = X.to_numpy()

    Y = df.iloc[:, -1]
    Y = Y.to_numpy()

    alpha_init = [2, 10]
    truncated_k = 50
    bnp = BayesianNonparametricMixture(alpha_init, truncated_k)

    if not os.path.exists('../models/posteriors.pkl'):
        bnp.train(num_iterations=30, x_train=X)

    if os.path.exists('../models/posteriors.pkl'):
        with open('../models/posteriors.pkl', 'rb') as f:
            saved_posteriors = pickle.load(f)
            niw_posteriors = saved_posteriors['niw_posteriors']
            alpha_posteriors = saved_posteriors['alpha_posteriors']
            sticks = saved_posteriors['sticks_list']
            sticks = sticks[29][:]  # the sticks from all 30 iterations were saved; I want only the last one
            log_likelihoods = saved_posteriors['log_likelihoods']
            active_clusters = saved_posteriors['active_clusters']

    mixing_weights = np.zeros(truncated_k)

    for k in range(truncated_k):
        mixing_weights[k] = weight_posterior(k, sticks)

    if not os.path.exists('../models/instance_log_likelihood.pkl'):
        instance_probs = np.array(instance_log_likelihood(X, truncated_k, niw_posteriors, mixing_weights))
        with open('../models/instance_log_likelihood.pkl', 'wb') as f:
            pickle.dump(instance_probs, f)

    with open('../models/instance_log_likelihood.pkl', 'rb') as f:
        instance_probs = pickle.load(f)

    # flag the clusters that have less than 1% of the total mixing weight as the inactive, and store them

    print(f"active clusters = {active_clusters}")

    # na = 'non active'

    """
    if not os.path.exists('../models/na_indices.pkl'):
        na_instances = non_active_instances(X[0:300], niw_posteriors, sticks, active_clusters, truncated_k)
        with open('../models/na_indices.pkl', 'wb') as f:
            pickle.dump(na_instances, f)
    else:
        with open('../models/na_indices.pkl', 'rb') as f:
            na_instances = pickle.load(f)

    print(na_instances[0:30])
    """

    """
    cluster_probs = na_cluster_probs(instance_probs, na_instances)
    mixing_weights = na_mixing_weights(mixing_weights[:, np.newaxis], na_instances)  # (no_instances, 1)
    weighted_probs = cluster_probs * mixing_weights

    anomaly_scores = np.log(np.sum(weighted_probs, axis=0))  # sum over clusters
    true_anomalies = [index for index, element in enumerate(Y) if element == 1] # actual 492

    # 14241 (initially) ->
    for percentile in range(1, 5):
        threshold = np.percentile(anomaly_scores, percentile)
        anomalies = [idx for idx, x in enumerate(X) if anomaly_scores[idx] < threshold ]
        print(f"Flagging top {percentile}% of instances as anomalous ({len(anomalies)}). "
              f"Detected {anomaly_statistics(anomalies, true_anomalies)} out of {len(true_anomalies)}")
    """