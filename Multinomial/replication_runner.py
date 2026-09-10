import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from data_generation import generate_smnp_data
from smnp_with_diagnostics import run_smnp_with_diagnostics


def _save_dataset(path, y, X, true_params):
    payload = {'y': np.asarray(y), 'X': np.asarray(X)}
    for k, v in true_params.items():
        if k in ('y', 'X'):
            continue
        payload[k] = np.asarray(v)
    np.savez(path, **payload)


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    y = d['y']
    X = d['X']
    true_params = {k: d[k] for k in d.files if k not in ('y', 'X')}

    if 'g_dist' in true_params:
        true_params['g_dist'] = str(true_params['g_dist'])
    return y, X, true_params


def _save_full_result(result_dir, fit_name, result, save_g=False, save_draws=True):
    if not save_draws:
        return

    os.makedirs(result_dir, exist_ok=True)
    samples_list = result.get('samples_list')
    if samples_list is not None:
        for ci, chain in enumerate(samples_list):
            arrays = {}
            for k, v in chain.items():
                if k == 'g' and not save_g:
                    continue
                arrays[k] = np.asarray(v)
            np.savez_compressed(
                os.path.join(result_dir, f"{fit_name}_chain{ci}.npz"), **arrays)


def _flatten_metric_dicts(res, fit_name):
    out = {}
    out[f'beta_rmse__{fit_name}'] = res.get('beta_rmse')
    out[f'delta_rmse__{fit_name}'] = res.get('delta_rmse')
    out[f'computation_time__{fit_name}'] = res.get('computation_time')

    derived = res.get('derived_quantities')
    if derived:
        for k, v in derived.items():
            if v is None or np.isscalar(v):
                out[f'derived__{k}__{fit_name}'] = v

    for mkey in ('test_metrics', 'train_metrics', 'train_conditional_metrics', 'train_marginal_metrics',
                 'test_marginal_metrics'):
        md = res.get(mkey) or {}
        for k, v in md.items():
            if v is None or np.isscalar(v):
                out[f'{mkey}__{k}__{fit_name}'] = v
    return out


def _build_replicate_row(r, seed, fits, cell_id=None):
    row = {'replicate': r, 'seed': seed}
    if cell_id:
        row.update(cell_id)
    for fit_name, res in fits.items():
        row.update(_flatten_metric_dicts(res, fit_name))
    return row


def run_one_replicate(r, run_dir, seed, N, j, p, beta_true, delta_true, gamma_true, Psi_minus_true, g_dist,
                      constrain_skewness, run_kwargs, test_size=0.3, split_seed=42, save_g=True, skew_id=None,
                      save_draws=True):
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_rng = np.random.default_rng(seed)
    delta_zero = np.zeros_like(delta_true)

    y_smnp, X_smnp, tp_smnp = generate_smnp_data(N, j, p, beta_true, delta_true, gamma_true, Psi_minus_true,
                                                 include_skewness=True, rng=data_rng, g_dist=g_dist)
    y_mnp, X_mnp, tp_mnp = generate_smnp_data(N, j, p, beta_true, delta_zero, gamma_true, Psi_minus_true,
                                              include_skewness=False, rng=data_rng, g_dist=g_dist)

    xtrain_smnp, xtest_smnp, ytrain_smnp, ytest_smnp = train_test_split(X_smnp, y_smnp, test_size=test_size,
                                                                        random_state=split_seed, stratify=y_smnp)
    xtrain_mnp, xtest_mnp, ytrain_mnp, ytest_mnp = train_test_split(X_mnp, y_mnp, test_size=test_size,
                                                                    random_state=split_seed, stratify=y_mnp)

    fits = {}

    # This is the first experiment: parameter recovery (we use the complete dataset)
    fits['smnp_on_smnp_recov'] = run_smnp_with_diagnostics(X_smnp, y_smnp, X_smnp, y_smnp, j, p, beta_true=beta_true,
                                                           delta_true=delta_true, true_params=tp_smnp,
                                                           experiment_name="smnp_on_smnp_recov", is_simulated_data=True,
                                                           output_dir=results_dir, include_skewness=True, g_dist=g_dist,
                                                           constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "smnp_on_smnp_recov", fits['smnp_on_smnp_recov'], save_g=save_g,
                      save_draws=save_draws)

    fits['mnp_on_smnp_recov'] = run_smnp_with_diagnostics(X_smnp, y_smnp, X_smnp, y_smnp, j, p, beta_true=beta_true,
                                                          delta_true=None, true_params=tp_smnp,
                                                          experiment_name="mnp_on_smnp_recov", is_simulated_data=True,
                                                          output_dir=results_dir, include_skewness=False, g_dist=g_dist,
                                                          constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "mnp_on_smnp_recov", fits['mnp_on_smnp_recov'], save_g=save_g, save_draws=save_draws)

    fits['smnp_on_mnp_recov'] = run_smnp_with_diagnostics(X_mnp, y_mnp, X_mnp, y_mnp, j, p, beta_true=beta_true,
                                                          delta_true=delta_zero, true_params=tp_mnp,
                                                          experiment_name="smnp_on_mnp_recov", is_simulated_data=True,
                                                          output_dir=results_dir, include_skewness=True, g_dist=g_dist,
                                                          constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "smnp_on_mnp_recov", fits['smnp_on_mnp_recov'], save_g=save_g, save_draws=save_draws)

    fits['mnp_on_mnp_recov'] = run_smnp_with_diagnostics(X_mnp, y_mnp, X_mnp, y_mnp, j, p, beta_true=beta_true,
                                                         delta_true=None, true_params=tp_mnp,
                                                         experiment_name="mnp_on_mnp_recov", is_simulated_data=True,
                                                         output_dir=results_dir, include_skewness=False, g_dist=g_dist,
                                                         constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "mnp_on_mnp_recov", fits['mnp_on_mnp_recov'], save_g=save_g, save_draws=save_draws)

    #  This is the second experiment: prediction (we fit on train data and score on test data)
    fits['smnp_on_smnp'] = run_smnp_with_diagnostics(xtrain_smnp, ytrain_smnp, xtest_smnp, ytest_smnp, j, p,
                                                     beta_true=beta_true, delta_true=delta_true, true_params=tp_smnp,
                                                     experiment_name="smnp_on_smnp", is_simulated_data=True,
                                                     output_dir=results_dir, include_skewness=True, g_dist=g_dist,
                                                     constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "smnp_on_smnp", fits['smnp_on_smnp'], save_g=save_g, save_draws=save_draws)

    fits['mnp_on_smnp'] = run_smnp_with_diagnostics(xtrain_smnp, ytrain_smnp, xtest_smnp, ytest_smnp, j, p,
                                                    beta_true=beta_true, delta_true=None, true_params=tp_smnp,
                                                    experiment_name="mnp_on_smnp", is_simulated_data=True,
                                                    output_dir=results_dir, include_skewness=False, g_dist=g_dist,
                                                    constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "mnp_on_smnp", fits['mnp_on_smnp'], save_g=save_g, save_draws=save_draws)

    fits['smnp_on_mnp'] = run_smnp_with_diagnostics(xtrain_mnp, ytrain_mnp, xtest_mnp, ytest_mnp, j, p,
                                                    beta_true=beta_true, delta_true=delta_zero, true_params=tp_mnp,
                                                    experiment_name="smnp_on_mnp", is_simulated_data=True,
                                                    output_dir=results_dir, include_skewness=True, g_dist=g_dist,
                                                    constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "smnp_on_mnp", fits['smnp_on_mnp'], save_g=save_g, save_draws=save_draws)

    fits['mnp_on_mnp'] = run_smnp_with_diagnostics(xtrain_mnp, ytrain_mnp, xtest_mnp, ytest_mnp, j, p,
                                                   beta_true=beta_true, delta_true=None, true_params=tp_mnp,
                                                   experiment_name="mnp_on_mnp", is_simulated_data=True,
                                                   output_dir=results_dir, include_skewness=False, g_dist=g_dist,
                                                   constrain_skewness=constrain_skewness, **run_kwargs)
    _save_full_result(results_dir, "mnp_on_mnp", fits['mnp_on_mnp'], save_g=save_g, save_draws=save_draws)

    cell_id = {'N': N, 'g_dist': g_dist, 'constrain_skewness': constrain_skewness}
    if skew_id is not None:
        cell_id['skew_id'] = skew_id
    row = _build_replicate_row(r, seed, fits, cell_id=cell_id)
    pd.DataFrame([row]).to_csv(os.path.join(results_dir, "replicate_metrics.csv"), index=False)

    return row


def run_replication_study(R, N, j, p, beta_true, delta_true, gamma_true, Psi_minus_true, g_dist, constrain_skewness,
                          run_kwargs, output_dir, base_seed=1000, test_size=0.3, split_seed=42, save_g=True,
                          resume=True, skew_id=None, save_draws=True):
    os.makedirs(output_dir, exist_ok=True)

    seed_seqs = np.random.SeedSequence(base_seed).spawn(R)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seqs]

    pd.DataFrame({'replicate': np.arange(R), 'seed': seed_ints}).to_csv(
        os.path.join(output_dir, "seed_manifest.csv"), index=False)

    rows = []
    for r in range(R):
        run_dir = os.path.join(output_dir, f"run_{r:03d}")
        rep_csv = os.path.join(run_dir, "results", "replicate_metrics.csv")

        if resume and os.path.exists(rep_csv):
            print(f"[replicate {r + 1}/{R}] found existing results --- loading, skipping fit")
            rows.append(pd.read_csv(rep_csv).iloc[0].to_dict())
            continue

        print(f"\n{'=' * 72}\n replicate {r + 1}/{R}  (seed={seed_ints[r]}) --- {run_dir}\n{'=' * 72}")
        row = run_one_replicate(r, run_dir, seed_ints[r], N, j, p, beta_true, delta_true, gamma_true, Psi_minus_true,
                                g_dist, constrain_skewness, run_kwargs, test_size=test_size, split_seed=split_seed,
                                save_g=save_g, skew_id=skew_id, save_draws=save_draws)
        rows.append(row)

    df = pd.DataFrame(rows)

    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    df.to_csv(os.path.join(agg_dir, "all_replicates.csv"), index=False)

    summary = summarize_replicates(df)
    summary.to_csv(os.path.join(agg_dir, "summary.csv"))

    print("\nAGGREGATE SUMMARY (head):")
    with pd.option_context('display.max_rows', 30, 'display.width', 160):
        print(summary.round(4).head(30))

    return df, summary


def summarize_replicates(df):
    skip = {'replicate', 'seed', 'N', 'g_dist', 'skew_id', 'constrain_skewness'}
    metric_cols = [c for c in df.columns if
                   c not in skip and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    out = {}
    for c in metric_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        sd = s.std(ddof=1) if len(s) > 1 else np.nan
        out[c] = {'mean': s.mean(), 'sd': sd, 'q2.5': s.quantile(0.025), 'median': s.median(),
                  'q97.5': s.quantile(0.975), 'mcse': (sd / np.sqrt(len(s))) if len(s) > 1 else np.nan, 'n': len(s)}
    return pd.DataFrame(out).T


def paired_difference(df, metric, model_a, model_b, data, j=None):
    col_a = f'test_metrics__{metric}__{model_a}_on_{data}'
    col_b = f'test_metrics__{metric}__{model_b}_on_{data}'
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"missing columns: {col_a} and/or {col_b}. "
                       f"Available test_metrics columns: "
                       f"{[c for c in df.columns if c.startswith('test_metrics__')][:12]} ...")
    diff = df[col_a] - df[col_b]
    diff.index = df['replicate'] if 'replicate' in df.columns else diff.index
    return diff


def summarize_paired_difference(df, metric, model_a, model_b, data):
    d = paired_difference(df, metric, model_a, model_b, data).dropna()
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    return {'metric': metric, 'contrast': f'{model_a}-{model_b} on {data} data', 'n': n, 'mean_diff': d.mean(),
            'sd': sd, 'mcse': (sd / np.sqrt(n)) if n > 1 else np.nan, 'q2.5': d.quantile(0.025),
            'q97.5': d.quantile(0.975), 'P(diff>0)': float(np.mean(d > 0))}


def per_class_recall_table(df, model_a='smnp', model_b='mnp', data='smnp', J=4):
    rows = []
    for c in range(1, J + 1):
        metric = f'recall_class_{c}'
        col_a = f'test_metrics__{metric}__{model_a}_on_{data}'
        col_b = f'test_metrics__{metric}__{model_b}_on_{data}'
        if col_a not in df.columns or col_b not in df.columns:
            continue
        rows.append({'class': c, **summarize_paired_difference(df, metric, model_a, model_b, data)})
    return pd.DataFrame(rows)
