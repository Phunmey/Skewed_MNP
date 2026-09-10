import numpy as np


def _sample_beta(self):
    """
    Sample stacked beta (k*(J-1),) and reshape to (J-1, k).
    the skewness term removed from the response is the outer product (g_i - mu_G) * delta_j.
    """
    psi_inv = self.Psi_inv_cached
    p, J_minus_one = self.p, self.J_minus_1

    if self.include_skewness:
        skew = (self.g - self.mu_G)[:, None] * self.delta
        W_adj = self.W - skew
    else:
        W_adj = self.W

    S2 = np.kron(psi_inv, self.XtX)

    weighted_W = np.dot(W_adj, psi_inv.T)
    M = np.dot(self.X.T, weighted_W)
    S1 = M.T.reshape(self.p_total)

    Vinv = S2 + self.Q_beta
    Vinv = 0.5 * (Vinv + Vinv.T)
    V_beta_hat = self._safe_inv(Vinv)
    b_beta = S1 + np.dot(self.Q_beta, self.mu_beta)
    mu_beta_hat = np.dot(V_beta_hat, b_beta)

    try:
        self.beta_vec = np.random.multivariate_normal( mu_beta_hat, V_beta_hat)
    except np.linalg.LinAlgError:
        #use a cholesky fallback when the internal SVD fails
        V_beta_hat = 0.5 * (V_beta_hat + V_beta_hat.T)

        L = np.linalg.cholesky(V_beta_hat)
        z = np.random.standard_normal(mu_beta_hat.shape[0])

        self.beta_vec = mu_beta_hat + L @ z


    self.beta = self.beta_vec.reshape((J_minus_one, p))
