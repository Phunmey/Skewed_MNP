import argparse, glob, os, re
import numpy as np
import pandas as pd
import arviz as az

from data_generation import generate_smnp_data
from run_sim import beta_true, delta_true, gamma_true, psi_minus_true, j, p

fits = {'smnp_on_smnp_recov': ('smnp', 'smnp', True), 'mnp_on_smnp_recov': ('smnp', 'mnp', False),
        'smnp_on_mnp_recov': ('mnp', 'smnp', True), 'mnp_on_mnp_recov': ('mnp', 'mnp', True)}


def infer_cell_info(cell_dir, N=None, g_dist=None):
    cell_dir = os.path.abspath(cell_dir)
    if N is None:
        m = re.search(r'result_N_(\d+)', os.path.basename(cell_dir))
        if m is None:
            raise ValueError('Could not infer N; pass --N.')
        N = int(m.group(1))
    if g_dist is None:
        parent = os.path.basename(os.path.dirname(cell_dir)).lower()
        if parent not in {'halfnormal', 'exponential'}:
            raise ValueError('Could not infer g_dist; pass g-dist.')
        g_dist = parent
    return N, g_dist


def norm_constants(X):
    X = np.asarray(X, float)
    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    for k in range(X.shape[1]):
        col = X[:, k]
        is_constant = np.std(col) < 1e-12
        is_binary = set(np.unique(col)).issubset({0.0, 1.0})
        if not is_constant and not is_binary:
            mean[k] = np.mean(col)
            std[k] = np.std(col)
            if std[k] < 1e-12:
                std[k] = 1.0
    return mean, std


def regenerate(seed, N, g_dist):
    rng = np.random.default_rng(int(seed))
    dz = np.zeros_like(delta_true)
    y_s, X_s, tp_s = generate_smnp_data(N, j, p, beta_true, delta_true, gamma_true, psi_minus_true,
                                        include_skewness=True, rng=rng, g_dist=g_dist, verbose=False)
    y_m, X_m, tp_m = generate_smnp_data(N, j, p, beta_true, dz, gamma_true, psi_minus_true, include_skewness=False,
                                        rng=rng, g_dist=g_dist, verbose=False)
    return {'smnp': (y_s, X_s, tp_s), 'mnp': (y_m, X_m, tp_m)}


def true_psi():
    m = j - 1
    P = np.zeros((m, m))
    P[0, 0] = 1.0
    P[0, 1:] = gamma_true
    P[1:, 0] = gamma_true
    P[1:, 1:] = psi_minus_true + np.outer(gamma_true, gamma_true)
    return P


def chain_no(path):
    m = re.search(r'_chain(\d+)\.npz$', os.path.basename(path))
    return int(m.group(1)) if m else 999


def load_chains(results_dir, fit, expected=4):
    paths = sorted(glob.glob(os.path.join(results_dir, f'{fit}_chain*.npz')), key=chain_no)
    if len(paths) < expected:
        return None, paths
    paths = paths[:expected]
    chains = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            chains.append({k: np.asarray(f[k]) for k in f.files})
    n = min(ch['beta'].shape[0] for ch in chains)
    common = set.intersection(*(set(ch) for ch in chains))
    return {k: np.stack([ch[k][:n] for ch in chains], axis=0) for k in common}, paths


def beta_to_raw(b, X_mean, X_std):
    out = np.asarray(b, float).copy()
    out[..., 1:] = b[..., 1:] / X_std[1:]
    out[..., 0] = b[..., 0] - np.sum(b[..., 1:] * (X_mean[1:] / X_std[1:]), axis=-1)
    return out


def diagnostics(draws):
    idata = az.from_dict(posterior={'x': np.asarray(draws, float)})
    rhat = np.asarray(az.rhat(idata, var_names=['x'])['x'].values).reshape(-1)
    bulk = np.asarray(az.ess(idata, var_names=['x'], method='bulk')['x'].values).reshape(-1)
    tail = np.asarray(az.ess(idata, var_names=['x'], method='tail')['x'].values).reshape(-1)
    return rhat, bulk, tail


def summarize(draws, truth, names, block, meta):
    draws = np.asarray(draws, float)
    truth = np.asarray(truth, float).reshape(-1)
    pooled = draws.reshape(draws.shape[0] * draws.shape[1], -1)
    mean = pooled.mean(axis=0)
    med = np.median(pooled, axis=0)
    lo = np.quantile(pooled, 0.025, axis=0)
    hi = np.quantile(pooled, 0.975, axis=0)
    rhat, bulk, tail = diagnostics(draws)
    rows = []
    for k, name in enumerate(names):
        err = mean[k] - truth[k]
        rows.append({
            **meta,
            'block': block,
            'parameter': name,
            'truth': truth[k],
            'post_mean': mean[k],
            'post_median': med[k],
            'q025': lo[k],
            'q975': hi[k],
            'bias': err,
            'squared_error': err ** 2,
            'coverage_95': bool(lo[k] <= truth[k] <= hi[k]),
            'ci_width': hi[k] - lo[k],
            'rhat': rhat[k],
            'ess_bulk': bulk[k],
            'ess_tail': tail[k]
        })
    return rows


def beta_names():
    return [f'beta_alt{a}_{"intercept" if k == 0 else f"x{k}"}'
            for a in range(2, j + 1) for k in range(p)]


def psi_free(psi_draws, P0):
    pairs = [(a, b) for a in range(j - 1) for b in range(a, j - 1) if not (a == 0 and b == 0)]
    draws = np.stack([psi_draws[..., a, b] for a, b in pairs], axis=-1)
    truth = np.array([P0[a, b] for a, b in pairs])
    names = [f'Psi_{a + 1}{b + 1}' for a, b in pairs]
    return draws, truth, names


def mix_constants(g_dist):
    if g_dist == 'halfnormal':
        v = 1.0 - 2.0 / np.pi
        mu3 = np.sqrt(2.0 / np.pi) * (4.0 / np.pi - 1.0)
    else:
        v, mu3 = 1.0, 2.0
    kappa3 = mu3 / v ** 1.5
    return v, kappa3


def derived(delta, psi, g_dist):
    v, kappa3 = mix_constants(g_dist)
    pd = np.diagonal(psi, axis1=-2, axis2=-1)
    lam = np.sqrt(v) * delta / np.sqrt(pd + v * delta ** 2)
    return lam ** 2, kappa3 * lam ** 3


def derived_truth(delta0, P0, g_dist):
    v, kappa3 = mix_constants(g_dist)
    lam = np.sqrt(v) * delta0 / np.sqrt(np.diag(P0) + v * delta0 ** 2)
    return lam ** 2, kappa3 * lam ** 3


def component_summary(df):
    keys = ['N', 'g_dist', 'fit', 'dgp', 'model', 'correctly_specified', 'block', 'parameter']
    out = []
    for vals, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, vals))
        row.update({
            'n_replicates': g['replicate'].nunique(),
            'truth': g['truth'].iloc[0],
            'mean_estimate': g['post_mean'].mean(),
            'mc_bias': g['bias'].mean(),
            'mc_rmse': np.sqrt(g['squared_error'].mean()),
            'coverage_95': g['coverage_95'].mean(),
            'mean_ci_width': g['ci_width'].mean(),
            'mean_rhat': g['rhat'].mean(),
            'max_rhat': g['rhat'].max(),
            'median_ess_bulk': g['ess_bulk'].median(),
            'min_ess_bulk': g['ess_bulk'].min(),
            'median_ess_tail': g['ess_tail'].median()
        })
        out.append(row)
    return pd.DataFrame(out)


def block_summary(df):
    keys = ['N', 'g_dist', 'fit', 'dgp', 'model', 'correctly_specified', 'block']
    out = []
    for vals, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, vals))
        row.update({
            'n_replicates': g['replicate'].nunique(),
            'n_components': g['parameter'].nunique(),
            'mc_rmse_pool': np.sqrt(g['squared_error'].mean()),
            'mean_absolute_bias': g['bias'].abs().mean(),
            'mean_coverage_95': g['coverage_95'].mean(),
            'mean_ci_width': g['ci_width'].mean(),
            'max_rhat': g['rhat'].max(),
            'min_ess_bulk': g['ess_bulk'].min()
        })
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-dir', required=True)
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--g-dist', choices=['halfnormal', 'exponential'], default=None)
    ap.add_argument('--expected-chains', type=int, default=4)
    args = ap.parse_args()

    cell_dir = os.path.abspath(args.cell_dir)
    N, g_dist = infer_cell_info(cell_dir, args.N, args.g_dist)
    seeds = pd.read_csv(os.path.join(cell_dir, 'seed_manifest.csv'))
    P0 = true_psi()
    dz = np.zeros_like(delta_true)

    rows, skipped = [], []
    for _, sr in seeds.iterrows():
        r, seed = int(sr['replicate']), int(sr['seed'])
        run_dir = os.path.join(cell_dir, f'run_{r:03d}')
        results_dir = os.path.join(run_dir, 'results')
        if not os.path.isdir(results_dir):
            skipped.append((r, 'results directory missing'))
            continue

        regen = None
        for fit, (dgp, model, correct) in fits.items():
            chains, paths = load_chains(results_dir, fit, args.expected_chains)
            if chains is None:
                skipped.append((r, f'{fit}: {len(paths)}/{args.expected_chains} chains'))
                continue

            if regen is None:
                regen = regenerate(seed, N, g_dist)

            norm_path = os.path.join(results_dir, fit, 'normalization_info.npz')
            if os.path.exists(norm_path):
                with np.load(norm_path, allow_pickle=False) as f:
                    Xm, Xs = np.asarray(f['X_mean']), np.asarray(f['X_std'])
                norm_source = 'saved'
            else:
                Xm, Xs = norm_constants(regen[dgp][1])
                norm_source = 'reconstructed_from_seed'

            meta = {'replicate': r, 'seed': seed, 'N': N, 'g_dist': g_dist, 'fit': fit, 'dgp': dgp, 'model': model,
                    'correctly_specified': correct, 'normalization_source': norm_source}

            braw = beta_to_raw(chains['beta'], Xm, Xs)
            rows += summarize(braw, np.asarray(beta_true).reshape(j - 1, p), beta_names(), 'beta', meta)

            pdraw, ptruth, pnames = psi_free(chains['Psi'], P0)
            rows += summarize(pdraw, ptruth, pnames, 'Psi', meta)

            if model == 'smnp':
                d0 = np.asarray(delta_true) if dgp == 'smnp' else dz
                dnames = [f'delta_alt{a}' for a in range(2, j + 1)]
                rows += summarize(chains['delta'], d0, dnames, 'delta', meta)

                lsq, msk = derived(chains['delta'], chains['Psi'], g_dist)
                lsq0, msk0 = derived_truth(d0, P0, g_dist)
                rows += summarize(lsq, lsq0, [f'lambda_sq_alt{a}' for a in range(2, j + 1)], 'lambda_sq', meta)
                rows += summarize(msk, msk0, [f'marginal_skewness_alt{a}' for a in range(2, j + 1)],
                                  'marginal_skewness', meta)

        print(f'processed run_{r:03d}')

    outdir = os.path.join(cell_dir, 'postprocess_recovery')
    os.makedirs(outdir, exist_ok=True)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(outdir, 'recovery_by_replicate_long.csv'), index=False)
    component_summary(long).to_csv(os.path.join(outdir, 'recovery_component_summary.csv'), index=False)
    block_summary(long).to_csv(os.path.join(outdir, 'recovery_block_summary.csv'), index=False)
    pd.DataFrame(skipped, columns=['replicate', 'reason']).to_csv(
        os.path.join(outdir, 'skipped_or_incomplete.csv'), index=False)

    print('\nSaved:')
    print(os.path.join(outdir, 'recovery_by_replicate_long.csv'))
    print(os.path.join(outdir, 'recovery_component_summary.csv'))
    print(os.path.join(outdir, 'recovery_block_summary.csv'))
    print(os.path.join(outdir, 'skipped_or_incomplete.csv'))


if __name__ == '__main__':
    main()
