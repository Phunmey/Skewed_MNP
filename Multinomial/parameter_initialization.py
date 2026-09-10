import numpy as np
from scipy.linalg import inv


def _default_priors(self):
    priors = {
        'mu_beta': np.zeros(self.p_total),
        'Q_beta': 0.25 * np.eye(self.p_total),   # beta ~ N(0, 4I)

        'mu_delta': np.zeros(self.J_minus_1),
        'Q_delta': 1.0 * np.eye(self.J_minus_1)  # delta ~ N(0, I)
    }

    if self.J_minus_2 > 0:
        q = self.J_minus_2

        kappa_0 = 5.0
        kappa = q + 6

        target_E_Phi = 1.0 / (1.0 + 1.0 / kappa_0)
        D_scale = (kappa - q - 1.0) * target_E_Phi

        priors.update({
            'mu_gamma': np.zeros(q),
            'kappa_0': kappa_0,
            'kappa': kappa,
            'D_wishart': D_scale * np.eye(q),
        })

    return priors


def _safe_inv(self, A, regularization=1e-8):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    A = 0.5 * (A + A.T)

    U, s, Vh = np.linalg.svd(A)
    s_reg = np.maximum(s, regularization)

    A_inv = (Vh.T * (1.0 / s_reg)) @ U.T

    return 0.5 * (A_inv + A_inv.T)



def _draw_g(self, n):
    g_dist = getattr(self, 'g_dist', 'halfnormal')
    if g_dist == 'halfnormal':
        return np.abs(np.random.randn(n))
    elif g_dist == 'exponential':
        return np.random.exponential(scale=1.0, size=n)
    else:
        raise ValueError(f"Unknown g_dist '{g_dist}'.")


def _initialize_parameters(self):
    self.W = self._initialize_W()

    self.beta_vec = np.random.multivariate_normal(self.mu_beta, inv(self.Q_beta))
    self.beta = self.beta_vec.reshape((self.J_minus_1, self.p))

    if self.include_skewness:
        raw_draw = np.random.multivariate_normal(self.mu_delta, inv(self.Q_delta))
        if getattr(self, 'constrain_skewness', False):
            raw_draw[0] = abs(raw_draw[0])
            self.delta = raw_draw
        else:
            self.delta = raw_draw
        self.g = _draw_g(self, self.N)
    else:
        self.delta = np.zeros(self.J_minus_1)
        self.g = np.zeros(self.N)

    if self.J_minus_2 > 0:
        max_cond, attempts = 1e3, 50
        for _ in range(attempts):
            self.Psi_minus = self._sample_initial_Psi_minus()
            self.gamma = np.random.multivariate_normal(
                self.mu_gamma, self.Psi_minus / self.kappa_0)
            Psi_try = self._construct_Psi()
            if np.linalg.cond(Psi_try) <= max_cond:
                break
    else:
        self.gamma = np.array([])
        self.Psi_minus = np.zeros((0, 0))

    self.Psi = self._construct_Psi()