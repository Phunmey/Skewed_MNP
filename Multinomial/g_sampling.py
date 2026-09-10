import numpy as np


def _sample_g(self):
    if not self.include_skewness:
        return

    psi_inv = self.Psi_inv_cached
    delta = self.delta
    mu_G = self.mu_G
    g_dist = getattr(self, 'g_dist', 'halfnormal')

    X_beta = np.dot(self.X, self.beta.T)
    r = (self.W - X_beta) + (mu_G * delta)

    Sinv_delta = psi_inv @ delta
    q = float(delta @ Sinv_delta)

    if g_dist == 'halfnormal':
        A = q + 1.0
        b = r @ Sinv_delta
    elif g_dist == 'exponential':
        lambda_G = getattr(self, 'lambda_G', 1.0)
        A = max(q, 1e-8)
        b = r @ Sinv_delta - lambda_G
    else:
        raise ValueError(f"Unknown g_dist '{g_dist}'. Use 'halfnormal' or 'exponential'.")


    var = 1.0 / A
    sd = np.sqrt(var)
    mean = b / A

    zeros = np.zeros(self.N)
    infs = np.full(self.N, np.inf)
    self.g = self._rtruncnorm_vec(mean, sd, zeros, infs)