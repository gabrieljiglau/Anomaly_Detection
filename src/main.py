import numpy as np
from pandas import read_feather
from pathlib import Path
from mixture import BayesianGaussianMixture
from src.pio import Loader
from src.utils import anomaly_statistics, log_likelihood_t, weight_posterior


def extract_probs(probs_map):
    probs = []
    inner_list = []
    for key in probs_map.keys():
        for element in probs_map[key]:
            inner_list.append(element)
        probs.append(inner_list)
    return np.array(probs)


if __name__ == '__main__':

    MAIN_PATH = Path(__file__).parent
    DATA_PATH = MAIN_PATH.parent / "datasets" / "processed" / "creditcard_standardized.feather"
    POSTERIORS_PATH = MAIN_PATH.parent / "models" / "posteriors.pkl"
    LOG_LIKELIHOOD_PATH = MAIN_PATH.parent / "models" / "instance_log_likelihood.pkl"

    df = read_feather(DATA_PATH)
    X = df.iloc[:, :-1]
    X = X.to_numpy()

    Y = df.iloc[:, -1]
    Y = Y.to_numpy()

    alpha_init = [2, 10]
    truncated_k = 50
    iterations = 30

    bnp = BayesianGaussianMixture(alpha_init, truncated_k)

    loader = Loader()
    niw_posteriors, alpha_posteriors, sticks, active_clusters, log_likelihoods = (
        loader.fully_load(POSTERIORS_PATH, bnp.train, 5, iterations, X))
    
    mixing_weights = np.zeros(truncated_k)
    for k in range(truncated_k):
        mixing_weights[k] = weight_posterior(sticks, k)

    cluster_probs_map = loader.fully_load(LOG_LIKELIHOOD_PATH,
                                          log_likelihood_t, 1, X, truncated_k,
                                          niw_posteriors, mixing_weights)

    # extract the probabilities from the dictionary if only the loader already has written the data
    cluster_probs = extract_probs(cluster_probs_map) if isinstance(cluster_probs_map, dict) else cluster_probs_map

    mixing_weights = mixing_weights[:, np.newaxis]
    weighted_probs = cluster_probs * mixing_weights

    anomaly_scores = np.log(np.sum(weighted_probs, axis=0))  # sum over clusters
    true_anomalies = [index for index, element in enumerate(Y) if element == 1]  # actual 492

    for percentile in range(1, 5):
        threshold = np.percentile(anomaly_scores, percentile)
        anomalies = [idx for idx, x in enumerate(X) if anomaly_scores[idx] < threshold]

        print(f"Flagging top {percentile}% of instances as anomalous ({len(anomalies)}). "
              f"Detected {anomaly_statistics(anomalies, true_anomalies)} out of {len(true_anomalies)}")