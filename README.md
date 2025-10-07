# AnomalyDetection

Identifying fraudulent [card transactions](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) using an 
implementation of the Bayesian Nonparametric Gaussian Mixture Model (BNP-GMM).


## Preamble

Centroid-based clustering needs a hyperparameter that controls the number of clusters to partition the data into. 
This is a reasonable approach when domain specific knowledge is available.
However, when little information is known about the true number of clusters, the task of finding such hyperparameter
is tedious. One approach is to use a nonparametric model that learns it from the dataset.

## Overview & Intuition

In a Bayesian setting for GMM, the prior over the weights is a Dirichlet distribution. If instead the prior was a 
Dirichlet process (a distribution over distributions), the effective number of clusters can be learned from the data,
thus the name 'nonparametric'.

In theory, such a model could have infinitely many clusters, albeit some of them would have tiny weights.
In practice, the number of clusters is approximated using a _truncated_ version (max clusters to consider).

There are different metaphors for the Dirichlet Process, but what was used here, in particular, was the 
'stick breaking process', where each cluster is given an initial weight from a 'stick' of length 1, based on sampling
from the Beta distribution, which has realizations inside [0, 1].

<img src="images/hierarchical_DP.png"  alt="Hierarchical DP flow" width=540/> [1]

More recently, the 'stick breaking prior' DP was used in Variational Encoders, in order to enhance their
representation capability [2].


## Results

After clustering using max _truncated_ clusters, it is expected that the ones with the lowest weight are fraudulent. Additionally, metrics like anomaly scores or threshold selection can be further employed in order to make more educated guesses.

### References
[1]: [Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. Journal of the American Statistical Association, 101(476), 1566–1581](https://doi.org/10.1198/016214506000000302)

[2]: [Nalisnick, E., & Smyth, P. (2017). Stick-Breaking Variational Autoencoders. In International Conference on Learning Representations (ICLR)](https://arxiv.org/abs/1605.06197)