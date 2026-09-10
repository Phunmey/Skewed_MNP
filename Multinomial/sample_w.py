import numpy as np


def _sample_W(self):
    J_minus_1 = self.J_minus_1
    y = self.y
    c = y - 2

    Omega = self.Psi_inv_cached
    if Omega is None:
        raise RuntimeError("_sample_W requires self.Psi_inv_cached to be set for Psi. ")

    S = np.dot(self.X, self.beta.T)
    if self.include_skewness:
        S = S + (self.g - self.mu_G)[:, None] * self.delta

    omega_diag = np.diag(Omega)
    sig_all = np.sqrt(1.0 / omega_diag)

    for j in range(J_minus_1):
        other = [l for l in range(J_minus_1) if l != j]
        sig_c = float(sig_all[j])

        if other:
            resid_other = self.W[:, other] - S[:, other]
            mu_c = S[:, j] - np.dot(resid_other, Omega[j, other]) / omega_diag[j]
        else:
            mu_c = S[:, j].copy()

        a = np.full(self.N, -np.inf)
        b = np.full(self.N, np.inf)

        mask_ref = (y == 1)
        mask_ch = (y >= 2) & (c == j)
        mask_ot = (y >= 2) & (c != j)

        b[mask_ref] = 0.0
        if other:
            other_max = self.W[:, other].max(axis=1)
        else:
            other_max = np.full(self.N, -np.inf)
        a[mask_ch] = np.maximum(0.0, other_max[mask_ch])

        rows_ot = np.flatnonzero(mask_ot)
        b[rows_ot] = self.W[rows_ot, c[rows_ot]]

        draw = self._rtruncnorm_vec(mu_c, sig_c, a, b)
        outside = (~np.isfinite(draw)) | (draw <= a) | (draw >= b)
        if np.any(outside):
            rows = np.flatnonzero(outside)[:5]
            raise RuntimeError(
                f"Truncated-normal draw hit or escaped its interval for "
                f"component j={j} at rows {rows.tolist()}: "
                f"draw={np.round(draw[rows], 12).tolist()}, "
                f"a={np.round(a[rows], 12).tolist()}, "
                f"b={np.round(b[rows], 12).tolist()}. This puts W on/outside "
                f"the boundary of R(y); check the tail branch of "
                f"_rtruncnorm_vec."
            )

        self.W[:, j] = draw


def check_cone(self, raise_on_fail=True):
    y = self.y
    c = y - 2
    W = self.W

    bad = np.zeros(self.N, dtype=bool)

    ref = (y == 1)
    if np.any(ref):
        bad[ref] = np.any(W[ref] >= 0.0, axis=1)

    nonref = np.flatnonzero(y >= 2)
    if nonref.size:
        chosen_cols = c[nonref]
        W_nonref = W[nonref, :]
        rows = np.arange(nonref.size)
        win = W_nonref[rows, chosen_cols]

        rivals = W_nonref.copy()
        rivals[rows, chosen_cols] = -np.inf
        rival_max = np.max(rivals, axis=1)

        bad[nonref] |= (win <= 0.0)
        bad[nonref] |= (rival_max >= win)

    n_bad = int(np.count_nonzero(bad))
    if n_bad and raise_on_fail:
        rows = np.flatnonzero(bad)[:5]
        raise RuntimeError(
            f"W left the cone R(y) for {n_bad}/{self.N} observations. "
            f"First offending rows {rows.tolist()}, y={y[rows].tolist()}, "
            f"W={np.round(W[rows], 6).tolist()}. A truncated-normal draw "
            f"escaped its interval; check the tail branch of _rtruncnorm_vec."
        )
    return n_bad