"""
Utilities for generating and working with datasets
"""

import numpy as np


def generate_simulated_dataset(n=1e6, p=0.5, shuffle=True):
    """
    Generate a simulated dataset of n examples with proportion p coming from the
    positive class.
    """
    n_pos = int(n * p)
    n_neg = n - n_pos
    # The weights and means are set such that the compnents are \sigma apart
    X_pos = np.random.multivariate_normal(mean=(0.5, 0.5), cov=np.eye(2), size=n_pos)
    y_pos = np.ones(n_pos)
    X_neg = np.random.multivariate_normal(mean=(-0.5, 0.5), cov=np.eye(2), size=n_neg)
    y_neg = np.zeros(n_neg)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    if shuffle:
        p = np.random.permutation(n)
        X, y = X[p], y[p]
    return X, y
