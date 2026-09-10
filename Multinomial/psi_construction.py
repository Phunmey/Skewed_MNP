import numpy as np
from scipy.linalg import inv
from scipy import stats
from scipy.stats import wishart


def _extract_u1_u_minus1(self):
    X_beta = np.dot(self.X, self.beta.T)
    if self.include_skewness:
        skew = (self.g - self.mu_G)[:, None] * self.delta
    else:
        skew = 0.0
    u = self.W - X_beta - skew

    u1 = u[:, 0]
    u_minus_1 = u[:, 1:] if self.J_minus_1 > 1 else np.zeros((self.N, 0))
    return u1, u_minus_1


def _sample_gamma(self):
    if self.J_minus_2 == 0:
        return

    u1, u_minus_1 = self._extract_u1_u_minus1()
    kappa_n = float(self.kappa_0) + np.sum(u1 ** 2)
    m_n = ((float(self.kappa_0) * self.mu_gamma) + (u_minus_1.T @ u1)) / kappa_n
    self.gamma = np.random.multivariate_normal(m_n, self.Psi_minus / kappa_n)


def _sample_initial_Psi_minus(self):
    if self.J_minus_2 == 1:
        scale = float(self.D_wishart[0, 0])
        sample = stats.invgamma.rvs(a=self.kappa / 2.0, scale=scale / 2.0)
        return np.array([[sample]])
    else:
        return inv(wishart.rvs(df=self.kappa, scale=inv(self.D_wishart)))



def _sample_Psi_minus(self):
    if self.J_minus_2 == 0:
        return

    u1, u_minus_1 = self._extract_u1_u_minus1()
    r = u_minus_1 - np.outer(u1, self.gamma)
    residual_scatter = r.T @ r

    gamma_diff = self.gamma - self.mu_gamma
    gamma_prior_scatter = (float(self.kappa_0) * np.outer(gamma_diff, gamma_diff))

    nu_post = self.kappa + self.N + 1
    D_post = (self.D_wishart + residual_scatter + gamma_prior_scatter)
    D_post = 0.5 * (D_post + D_post.T)

    scale_post = self._safe_inv(D_post)
    V_inv_sample = wishart.rvs(df=nu_post, scale=scale_post)
    V_inv_sample = np.atleast_2d(V_inv_sample)
    self.Psi_minus = self._safe_inv(V_inv_sample)
    self.Psi_minus = 0.5 * (self.Psi_minus + self.Psi_minus.T)



def _construct_Psi(self):
    if self.J_minus_2 == 0:
        return np.array([[1.0]])

    Psi = np.zeros((self.J_minus_1, self.J_minus_1))
    Psi[0, 0] = 1.0
    Psi[0, 1:] = self.gamma
    Psi[1:, 0] = self.gamma
    Psi[1:, 1:] = self.Psi_minus + np.outer(self.gamma, self.gamma)
    return Psi
