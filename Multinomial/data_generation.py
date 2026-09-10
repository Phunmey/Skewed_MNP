import numpy as np
from scipy.stats import skew

def generate_smnp_data(N, j, p, beta_true, delta_true, gamma_true, Psi_minus_true,
                       include_skewness=True, rng=None, g_dist='halfnormal', label=None, verbose=True):

    rng = np.random.default_rng(42) if rng is None else rng
    j_minus_1 = j - 1

    if p != 4:
        raise ValueError(
            "generate_smnp_data currently assumes p=4: intercept + "
            "two continuous covariates + one binary covariate. "
            "Either set p=4 or generalize the X-generation block."
        )

    if j == 2:
        Psi_true = np.array([[1.0]])
    else:
        Psi_true = np.zeros((j_minus_1, j_minus_1))
        Psi_true[0, 0] = 1.0
        Psi_true[0, 1:] = gamma_true
        Psi_true[1:, 0] = gamma_true
        Psi_true[1:, 1:] = Psi_minus_true + np.outer(gamma_true, gamma_true)

    X_raw = np.zeros((N, p))

    X_raw[:, 0] = 1.0
    X_raw[:, 1] = rng.normal(0, 1.2, size=N)
    X_raw[:, 2] = rng.normal(0, 1.0, size=N)

    binary_col = rng.binomial(1, 0.3, size=N)
    X_raw[:, 3] = binary_col
    X = X_raw

    beta_matrix = np.asarray(beta_true).reshape((j_minus_1, p))

    if include_skewness:
        if g_dist == 'halfnormal':
            g_data = np.abs(rng.standard_normal(N))
            mu_G = np.sqrt(2.0 / np.pi)
        else:
            g_data = rng.exponential(scale=1.0, size=N)
            mu_G = 1.0
        delta = np.asarray(delta_true, dtype=float)
    else:
        g_data = np.zeros(N)
        mu_G = 0.0
        delta = np.zeros(j_minus_1)

    y = np.zeros(N, dtype=int)
    W_data = np.zeros((N, j_minus_1))

    for i in range(N):
        mean_i = np.dot(X[i], beta_matrix.T)
        if include_skewness:
            mean_i = mean_i + (delta * (g_data[i] - mu_G))
        W_i = rng.multivariate_normal(mean_i, Psi_true)
        W_data[i] = W_i

        if np.all(W_i <= 0):
            y[i] = 1
        else:
            y[i] = int(np.argmax(W_i)) + 2

    true_params = {
        'beta': np.asarray(beta_true), 'delta': delta_true,
        'gamma': gamma_true, 'Psi_minus': Psi_minus_true, 'Psi': Psi_true,
        'W': W_data, 'g': g_data, 'g_dist': g_dist, 'mu_G': mu_G, 'X': X, 'y': y
    }

    if label is None:
        label = "SMNP" if include_skewness else "MNP"

    if verbose:
        for alt in range(j - 1):
            W_alt = true_params['W'][:, alt]
            print(f"[{label}] Alternative {alt + 2}: Skewness = {skew(W_alt):.3f}")

    return y, X, true_params