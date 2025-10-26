import numpy as np
from pandas import read_csv
from pathlib import Path
from src.mixture.bnp_gmm import BayesianGaussianMixture, weight_posterior, log_likelihood_t
from src.pio.operations import Loader
from src.utils import anomaly_statistics

if __name__ == '__main__':

    MAIN_PATH = Path(__file__).parent
    DATA_PATH = MAIN_PATH.parent / "datasets" / "iris_numeric.csv"
    POSTERIORS_PATH = MAIN_PATH.parent / "models" / "posteriors1.pkl"
    LOG_LIKELIHOOD_PATH = MAIN_PATH.parent / "models" / "instance_log_likelihood1.pkl"

    # df = read_feather('../datasets/processed/creditcard_standardized.feather')
    df = read_csv(DATA_PATH)
    X = df.iloc[0:10, :-1]
    X = X.to_numpy()

    Y = df.iloc[:, -1]
    Y = Y.to_numpy()

    alpha_init = [2, 10]
    truncated_k = 3  # 50
    iterations = 3 # 30

    bnp = BayesianGaussianMixture(alpha_init, truncated_k)

    loader = Loader()
    niw_posteriors, alpha_posteriors, sticks, active_clusters, log_likelihoods = (
        loader.fully_load(POSTERIORS_PATH, bnp.train, 5, iterations, X))

    # sticks = sticks[iterations] # the sticks from all 30 iterations were saved; I only want the last one

    mixing_weights = np.zeros(truncated_k)
    for k in range(truncated_k):
        mixing_weights[k] = weight_posterior(k, sticks)

    cluster_probs_map = loader.fully_load(LOG_LIKELIHOOD_PATH,
                                          log_likelihood_t, 1, X, truncated_k,
                                          niw_posteriors, mixing_weights)

    # extract the probabilities from the dictionary
    cluster_probs = []
    inner_list = []
    for key in cluster_probs_map.keys():
        for element in cluster_probs_map[key]:
            inner_list.append(element)
        cluster_probs.append(inner_list)

    cluster_probs = np.array(cluster_probs)

    mixing_weights = mixing_weights[:, np.newaxis]
    weighted_probs = cluster_probs * mixing_weights

    anomaly_scores = np.log(np.sum(weighted_probs, axis=0))  # sum over clusters
    true_anomalies = [index for index, element in enumerate(Y) if element == 1]  # actual 492

    for percentile in range(1, 5):
        threshold = np.percentile(anomaly_scores, percentile)
        anomalies = [idx for idx, x in enumerate(X) if anomaly_scores[idx] < threshold]

        print(f"Flagging top {percentile}% of instances as anomalous ({len(anomalies)}). "
              f"Detected {anomaly_statistics(anomalies, true_anomalies)} out of {len(true_anomalies)}")
