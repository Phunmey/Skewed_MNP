import numpy as np

__all__ = [
    "mixing", "mixing_constants", "omega_from_draws", "lambda_from_draws",
    "gamma1_from_draws", "lambda_star_sq", "identified_set_slack",
    "derived_summary", "summarize_draws", "load_chain_files"
]

mixing = {
    "halfnormal": {
        "mu_G": np.sqrt(2.0 / np.pi),
        "v_G": 1.0 - 2.0 / np.pi,
        "mu3_G": np.sqrt(2.0 / np.pi) * (4.0 / np.pi - 1.0)
    },
    "exponential": {"mu_G": 1.0, "v_G": 1.0, "mu3_G": 2.0}
}
for _m in mixing.values():
    _m["kappa3_G"] = _m["mu3_G"] / _m["v_G"] ** 1.5

mixing["half-normal"] = mixing["halfnormal"]


def mixing_constants(g_dist):
    """Return (mu_G, v_G, kappa3_G) for the mixing distribution."""
    try:
        m = mixing[g_dist]
    except KeyError:
        raise ValueError(f"unknown g_dist {g_dist!r}; use 'halfnormal' or 'exponential'") from None
    return m["mu_G"], m["v_G"], m["kappa3_G"]


def _as_draws(delta, psi):
    delta = np.asarray(delta, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if delta.ndim == 2:
        delta = delta[None, ...]
    if psi.ndim == 3:
        psi = psi[None, ...]
    if delta.ndim != 3 or psi.ndim != 4:
        raise ValueError(f"expected delta (C, M, K) and psi (C, M, K, K); got {delta.shape} and {psi.shape}")
    if delta.shape[:2] != psi.shape[:2] or delta.shape[2] != psi.shape[2]:
        raise ValueError(f"delta {delta.shape} and psi {psi.shape} disagree")
    return delta, psi


def omega_from_draws(delta, psi, g_dist):
    delta, psi = _as_draws(delta, psi)
    _, v_G, _ = mixing_constants(g_dist)
    return psi + v_G * (delta[..., :, None] * delta[..., None, :])


def lambda_from_draws(delta, psi, g_dist):
    delta, psi = _as_draws(delta, psi)
    _, v_G, _ = mixing_constants(g_dist)
    omega_jj = np.einsum("cmjj-cmj", omega_from_draws(delta, psi, g_dist))
    return np.sqrt(v_G) * delta / np.sqrt(omega_jj)


def gamma1_from_draws(delta, psi, g_dist):
    _, _, kappa3 = mixing_constants(g_dist)
    return kappa3 * lambda_from_draws(delta, psi, g_dist) ** 3


def lambda_star_sq(delta, psi, g_dist):
    delta, psi = _as_draws(delta, psi)
    _, v_G, _ = mixing_constants(g_dist)
    t = v_G * np.einsum("cmj,cmjk,cmk-cm", delta, np.linalg.inv(psi), delta)
    return t / (1.0 + t)


def identified_set_slack(delta, psi, g_dist):
    delta, psi = _as_draws(delta, psi)
    return np.linalg.eigvalsh(psi).min(axis=-1)


def _q(x, lo=2.5, hi=97.5):
    a = np.asarray(x, dtype=float).reshape(-1)
    return (float(np.mean(a)), float(np.median(a)),
            float(np.percentile(a, lo)), float(np.percentile(a, hi)))


def derived_summary(delta, psi, g_dist, true_delta=None):
    delta, psi = _as_draws(delta, psi)
    _, v_G, kappa3 = mixing_constants(g_dist)
    C, M, K = delta.shape

    Omega = omega_from_draws(delta, psi, g_dist)
    lam = lambda_from_draws(delta, psi, g_dist)
    g1 = kappa3 * lam ** 3
    ls2 = lambda_star_sq(delta, psi, g_dist)
    slack = identified_set_slack(delta, psi, g_dist)

    out = {"g_dist": g_dist, "n_chains": C, "n_draws": M, "K": K, "v_G": v_G, "kappa3_G": kappa3}

    m, md, lo, hi = _q(ls2)
    out.update({"lambda_star_sq_mean": m, "lambda_star_sq_median": md,
                "lambda_star_sq_q025": lo, "lambda_star_sq_q975": hi,
                "lambda_star_sq_width": hi - lo,
                "lambda_star_mean": float(np.mean(np.sqrt(ls2)))})

    m, md, lo, hi = _q(slack)
    out.update({"psi_min_eig_mean": m, "psi_min_eig_q025": lo, "psi_min_eig_q975": hi})

    for j in range(K):
        p = f"j{j + 1}"
        m, md, lo, hi = _q(Omega[:, :, j, j])
        out.update({f"omega_{p}_mean": m, f"omega_{p}_q025": lo, f"omega_{p}_q975": hi})

        m, md, lo, hi = _q(np.abs(lam[:, :, j]))
        out.update({f"abs_lambda_{p}_mean": m, f"abs_lambda_{p}_q025": lo, f"abs_lambda_{p}_q975": hi})

        m, md, lo, hi = _q(np.abs(g1[:, :, j]))
        out.update({f"abs_gamma1_{p}_mean": m, f"abs_gamma1_{p}_q025": lo, f"abs_gamma1_{p}_q975": hi})

        m, md, lo, hi = _q(np.abs(delta[:, :, j]))
        out.update({f"abs_delta_{p}_mean": m, f"abs_delta_{p}_q025": lo, f"abs_delta_{p}_q975": hi})

        out[f"P_delta_positive_{p}"] = float((delta[:, :, j] > 0).mean())
        s = np.sign(delta[:, :, j])
        s[s == 0] = 1.0
        out[f"sign_flips_per_1k_{p}"] = float(
            np.mean(np.abs(np.diff(s, axis=-1)) > 0) * 1000.0)

        if true_delta is not None:
            t = abs(float(np.asarray(true_delta).reshape(-1)[j]))
            out[f"true_abs_delta_{p}"] = t
            out[f"abs_delta_covers_{p}"] = bool(
                out[f"abs_delta_{p}_q025"] <= t <= out[f"abs_delta_{p}_q975"])

    for a in range(K):
        for b in range(a + 1, K):
            prod = delta[:, :, a] * delta[:, :, b]
            out[f"delta_prod_j{a + 1}j{b + 1}_mean"] = float(prod.mean())
            out[f"P_delta_prod_positive_j{a + 1}j{b + 1}"] = float((prod > 0).mean())

            den = np.sqrt(Omega[:, :, a, a] * Omega[:, :, b, b])
            out[f"omega_corr_j{a + 1}j{b + 1}_mean"] = float((Omega[:, :, a, b] / den).mean())

    return out


def load_chain_files(paths):
    ds, ps = [], []
    for p in sorted(paths):
        z = np.load(p, allow_pickle=False)
        keys = {k.lower(): k for k in z.files}
        if "delta" not in keys:
            raise KeyError(f"{p}: no 'delta' array (found {z.files})")
        d = np.asarray(z[keys["delta"]], dtype=float)
        if "psi" in keys:
            P = np.asarray(z[keys["psi"]], dtype=float)
        elif "gamma" in keys and "psi_minus" in keys:
            gam = np.asarray(z[keys["gamma"]], dtype=float)
            phi = np.asarray(z[keys["psi_minus"]], dtype=float)
            M, Km1 = gam.shape
            K = Km1 + 1
            P = np.empty((M, K, K))
            P[:, 0, 0] = 1.0
            P[:, 0, 1:] = gam
            P[:, 1:, 0] = gam
            P[:, 1:, 1:] = phi + gam[:, :, None] * gam[:, None, :]
        else:
            raise KeyError(f"{p}: cannot form Psi (found {z.files})")
        ds.append(d)
        ps.append(P)

    n = min(d.shape[0] for d in ds)
    return (np.stack([d[:n] for d in ds]), np.stack([P[:n] for P in ps]))


def summarize_draws(paths, g_dist, true_delta=None, burn_in=0, thin=1):
    delta, psi = load_chain_files(paths)
    if burn_in:
        delta, psi = delta[:, burn_in:], psi[:, burn_in:]
    if thin > 1:
        delta, psi = delta[:, ::thin], psi[:, ::thin]
    return derived_summary(delta, psi, g_dist, true_delta=true_delta)