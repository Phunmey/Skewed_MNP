import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from parameter_initialization import (_initialize_parameters as _initialize_parameters_impl,
                                      _default_priors as _default_priors_impl, _safe_inv as _safe_inv_impl)
from w_initialization import _initialize_W as _initialize_W_impl
from psi_construction import (_construct_Psi as _construct_Psi_impl,
                              _sample_initial_Psi_minus as _sample_initial_Psi_minus_impl,
                              _sample_gamma as _sample_gamma_impl, _sample_Psi_minus as _sample_Psi_minus_impl,
                              _extract_u1_u_minus1 as _extract_U_E_impl)
from truncated_normal import _rtruncnorm_vec as _rtruncnorm_vec_impl
from sample_w import _sample_W as _sample_W_impl
from g_sampling import _sample_g as _sample_g_impl
from beta_sampling import _sample_beta as _sample_beta_impl
from delta import _sample_delta as _sample_delta_impl
from sampling_gibbs import sample as _sample_impl


class SMNPGibbsSampler:
    def __init__(self, y, X, J, p, prior_params=None, include_skewness=True, g_dist='halfnormal',
                 constrain_skewness=False):

        self.y = np.asarray(y)
        self.J = J
        self.N = len(self.y)
        self.J_minus_1 = J - 1
        self.J_minus_2 = max(0, J - 2)
        self.p = p
        self.p_total = self.p * self.J_minus_1
        self.include_skewness = include_skewness
        self.constrain_skewness = constrain_skewness and include_skewness
        self.g_structure = 'shared'
        self.g_dist = g_dist

        if g_dist == 'halfnormal':
            self.mu_G = np.sqrt(2.0 / np.pi)
        elif g_dist == 'exponential':
            self.mu_G = 1.0
        else:
            raise ValueError(f"Unknown g_dist '{g_dist}'. Use 'halfnormal' or 'exponential'.")

        self.sep = 1e-10

        if isinstance(X, pd.DataFrame):
            self.X_columns = list(X.columns)
            X_array = X.to_numpy(dtype=float)
        else:
            self.X_columns = None
            X_array = np.asarray(X, dtype=float).copy()

        self.X_mean = None
        self.X_std = None

        self.X = X_array.copy()
        self.XtX = self.X.T @ self.X

        expected_X_shape = (self.N, self.p)
        if X.shape != expected_X_shape:
            raise ValueError(f"X shape {X.shape} doesn't match expected {expected_X_shape}")

        if prior_params is None:
            prior_params = self._default_priors()

        self.mu_beta = prior_params['mu_beta']
        self.Q_beta = prior_params['Q_beta']
        self.mu_delta = prior_params['mu_delta']
        self.Q_delta = prior_params['Q_delta']

        if self.J_minus_2 > 0:
            self.mu_gamma = prior_params['mu_gamma']
            self.kappa_0 = prior_params['kappa_0']
            self.kappa = prior_params['kappa']
            self.D_wishart = prior_params['D_wishart']

        self.Psi_inv_cached = None
        self._initialize_parameters()

    def _default_priors(self):
        return _default_priors_impl(self)

    def _initialize_parameters(self):
        return _initialize_parameters_impl(self)

    def _initialize_W(self):
        return _initialize_W_impl(self)

    def _sample_initial_Psi_minus(self):
        return _sample_initial_Psi_minus_impl(self)

    def _construct_Psi(self):
        return _construct_Psi_impl(self)

    def _safe_inv(self, A, regularization=1e-8):
        return _safe_inv_impl(self, A, regularization=regularization)

    def _rtruncnorm_vec(self, mu, sigma, a, b):
        return _rtruncnorm_vec_impl(self, mu, sigma, a, b)

    def _sample_W(self):
        return _sample_W_impl(self)

    def _sample_g(self):
        return _sample_g_impl(self)

    def _sample_beta(self):
        return _sample_beta_impl(self)

    def _sample_delta(self):
        return _sample_delta_impl(self)

    def _sample_gamma(self):
        return _sample_gamma_impl(self)

    def _sample_Psi_minus(self):
        return _sample_Psi_minus_impl(self)

    def _extract_u1_u_minus1(self):
        return _extract_U_E_impl(self)

    def gibbs_step(self):
        self.Psi_inv_cached = self._safe_inv(self.Psi)
        self._sample_W()
        if self.include_skewness:
            self._sample_g()
            self._sample_delta()
        self._sample_beta()
        if self.J_minus_2 > 0:
            self._sample_gamma()
            self._sample_Psi_minus()
        self.Psi = self._construct_Psi()
        self.Psi_inv_cached = None

    def sample(self, n_samples, burn_in=100, thin=1, warm_start=50, verbose=True, store_g=True):
        return _sample_impl(self, n_samples, burn_in=burn_in, thin=thin, warm_start=warm_start, verbose=verbose,
                            store_g=store_g)
