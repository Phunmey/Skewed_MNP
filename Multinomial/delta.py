import numpy as np


def _sample_delta(self):
    if not self.include_skewness:
        return

    psi_inv = self.Psi_inv_cached
    Wdag = self.W - np.dot(self.X, self.beta.T)

    c = self.g - self.mu_G
    sum_c2 = float(np.sum(c ** 2))

    posterior_prec = sum_c2 * psi_inv + self.Q_delta
    posterior_prec = 0.5 * (posterior_prec + posterior_prec.T)

    psiinv_Wadj = np.dot(Wdag, psi_inv.T)
    b_delta = np.einsum('i,ij->j', c, psiinv_Wadj) + np.dot(self.Q_delta, self.mu_delta)
    V_delta = self._safe_inv(posterior_prec)
    mu_delta_hat = np.dot(V_delta, b_delta)

    if not getattr(self, 'constrain_skewness', False):
        self.delta = np.random.multivariate_normal(mu_delta_hat, V_delta)
        return

    d0 = float(self._rtruncnorm_vec(
        np.array([mu_delta_hat[0]]), np.sqrt(max(V_delta[0, 0], 1e-12)),
        np.array([0.0]), np.array([np.inf])
    )[0])

    if self.J_minus_1 == 1:
        self.delta = np.array([d0])
        return

    V00 = V_delta[0, 0]
    V10 = V_delta[1:, 0]
    V11 = V_delta[1:, 1:]
    cond_mean = mu_delta_hat[1:] + V10 * (d0 - mu_delta_hat[0]) / V00
    cond_cov = V11 - np.outer(V10, V10) / V00
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    cond_cov += 1e-10 * np.eye(cond_cov.shape[0])
    d_rest = np.random.multivariate_normal(cond_mean, cond_cov)

    self.delta = np.concatenate([[d0], d_rest])

    return