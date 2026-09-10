"""
This scripts generates the cloglog (asymmetric) and probit (symmetric) data respectively.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

true_beta = {
    'cloglog': np.array([1.5, -1.2, -3, 0.5, -0.5]),    # skewed / asymmetric DGP
    'probit':  np.array([1.5, 1, -5, 1.25, -0.25]),  # non-skewed / symmetric DGP
}

feature_names = ["x0", "x1", "x2", "x3", "x4"]


def generate_simulated_data(n, model='cloglog', seed=42):
    if model not in true_beta:
        raise ValueError("model must be 'cloglog' or 'probit'")
    beta = true_beta[model]
    rng = np.random.default_rng(seed)

    x0 = np.ones(n)
    x1 = rng.normal(0, 1, size=n)

    # x2, x3 are two dummies for a 3-level nominal categorical variable
    cat3 = rng.choice([0, 1, 2], size=n)
    x2 = (cat3 == 1).astype(int)
    x3 = (cat3 == 2).astype(int)
    x4 = rng.choice([0, 1], size=n)

    X = np.column_stack((x0, x1, x2, x3, x4))
    eta = X.dot(beta)

    if model == 'cloglog':
        p = 1 - np.exp(-np.exp(eta))
    else:
        p = norm.cdf(eta)

    y = rng.binomial(1, p)

    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    return df
