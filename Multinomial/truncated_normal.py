import numpy as np
from scipy.stats import truncnorm, norm


def _rtruncnorm_vec(self, mu, sigma, a, b):
    mu = np.asarray(mu, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), mu.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), mu.shape)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.broadcast_to(sigma, mu.shape) if sigma.ndim else np.full(mu.shape, float(sigma))

    delta = np.where(np.isfinite(a), (a - mu) / sigma, -np.inf)
    beta = np.where(np.isfinite(b), (b - mu) / sigma, np.inf)

    Fa = norm.cdf(delta)
    Fb = norm.cdf(beta)
    mass = Fb - Fa

    out = np.empty(mu.shape, dtype=float)

    ok = mass > 1e-10
    if np.any(ok):
        u = np.random.uniform(size=int(np.count_nonzero(ok)))
        p = Fa[ok] + u * mass[ok]
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        out[ok] = mu[ok] + sigma[ok] * norm.ppf(p)

    bad = ~ok
    if np.any(bad):
        out[bad] = truncnorm.rvs(delta[bad], beta[bad], loc=mu[bad], scale=sigma[bad])

    return out
