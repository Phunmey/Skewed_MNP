import argparse
import os


def limit_blas_threads(n=1):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(n))


limit_blas_threads(1)

import time
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from smnp_with_diagnostics import run_smnp_with_diagnostics, normalize_train_test
from replication_runner import (_build_replicate_row, _save_full_result, summarize_replicates, paired_difference,
                                per_class_recall_table)
from derived_quantities import lambda_from_draws, gamma1_from_draws


def reset_feature_types(df, categorical_cols=None, numeric_cols=None):
    df = df.copy()
    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype('category')

    if numeric_cols:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_real_data(data_path, feature_columns=None, target_column="Disposition",
                   nominal_cols=('Race', 'Arrival_transport'), ordinal_numeric_cols=('Acuity',),
                   binary_cols=('Gender',)):
    df1 = pd.read_csv(data_path, sep="\t")

    numeric_cols = (['Temperature', 'Heart rate', 'Resp.rate', 'O2 saturation',
                     'Systolic Blood Pressure (SBP)', 'Diastolic Blood Pressure (DBP)',
                     'Drugs', 'Pain', 'Age', 'Hours']
                    + list(ordinal_numeric_cols) + list(binary_cols))
    categorical_cols = [target_column] + list(nominal_cols)

    df = reset_feature_types(df1, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    y_original = df[target_column]
    # Ensure the target column is numeric
    if y_original.dtype.name == 'category':
        y_original = y_original.astype('int')

    if np.min(y_original) == 0:
        y = y_original + 1
    else:
        y = y_original

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    dummy_cols = [c for c in feature_columns if c in nominal_cols]
    base_cols = [c for c in feature_columns if c not in nominal_cols]

    X_base = df[base_cols].astype(float)

    dummy_ref_levels = {c: df[c].value_counts(dropna=True).idxmax() for c in dummy_cols}
    if dummy_cols:
        X_dummies_full = pd.get_dummies(df[dummy_cols], columns=dummy_cols, drop_first=False, prefix=dummy_cols)
        drop_cols = [f'{c}_{dummy_ref_levels[c]}' for c in dummy_cols]
        missing = [c for c in drop_cols if c not in X_dummies_full.columns]
        if missing:
            raise RuntimeError(f"Expected reference dummy columns {missing} not found in "
                               f"{list(X_dummies_full.columns)}; get_dummies column naming did not "
                               f"match dummy_ref_levels as assumed.")
        X_dummies = X_dummies_full.drop(columns=drop_cols).astype(float)
    else:
        X_dummies = pd.DataFrame(index=df.index)

    print("Predictor reference levels:")
    if dummy_ref_levels:
        for variable, reference in dummy_ref_levels.items():
            print(f"{variable}: {reference}")
    else:
        print("  none")

    X = pd.concat([X_base, X_dummies], axis=1)
    X.insert(0, 'intercept', 1.0)
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError("Missing/non-numeric values remain after preprocessing in columns: "
                         f"{bad_cols}. Handle these before fitting.")
    feature_names = list(X.columns)

    j = len(np.unique(y))
    p = X.shape[1]
    N = len(y)

    print(f"Choice alternatives: {sorted(np.unique(y))}")
    print(f"Number of alternatives (j): {j}")
    print(f"Number of features (p): {p}  [intercept + {len(base_cols)} base columns + "
          f"{X_dummies.shape[1]} dummy columns")
    print(f"Sample size: {len(y)}")
    print(f"Choice distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Feature columns: {feature_names}")

    return X, y, j, p, N, feature_names


def _resolve_g_dists(g_dist):
    if g_dist == "both":
        return ("halfnormal", "exponential")
    return (g_dist,)


def _g_label(g_dists):
    return "both" if len(g_dists) == 2 else tuple(g_dists)[0]


def _rep_seed(base_seed, rep_index):
    child = np.random.SeedSequence(base_seed).spawn(rep_index + 1)[rep_index]
    return int(child.generate_state(1)[0])


def _prediction_root(output_dir, g_dists):
    return os.path.join(output_dir, "prediction", _g_label(g_dists))


def _fit_real_model(X_train, y_train, X_test, y_test, j, p, output_dir, fit_name, include_skewness, g_dist, n_samples,
                    burn_in, n_chains, thin, store_g, n_cores=None, is_simulated_data=False):
    return run_smnp_with_diagnostics(X_train, y_train, X_test, y_test, j, p, beta_true=None, delta_true=None,
                                     true_params=None, experiment_name=fit_name, is_simulated_data=is_simulated_data,
                                     output_dir=output_dir, include_skewness=include_skewness, n_cores=n_cores,
                                     g_dist=g_dist, constrain_skewness=False, n_samples=n_samples, burn_in=burn_in,
                                     n_chains=n_chains, thin=thin, store_g=store_g)


def run_one_real_split(r, run_dir, split_seed, X, y, j, p, g_dists, test_size, n_samples, burn_in, n_chains, thin,
                       n_cores=None, save_draws=False, save_g=False):
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed,
                                                        stratify=y)

    fits = {}
    mnp_name = "mnp_on_real"
    fits[mnp_name] = _fit_real_model(X_train, y_train, X_test, y_test, j, p, output_dir=results_dir, fit_name=mnp_name,
                                     include_skewness=False, g_dist="halfnormal", n_samples=n_samples, burn_in=burn_in,
                                     n_chains=n_chains, thin=thin, store_g=False, n_cores=n_cores, is_simulated_data=False)

    _save_full_result(results_dir, mnp_name, fits[mnp_name], save_g=False, save_draws=save_draws)

    for gd in g_dists:
        smnp_name = f"smnp_{gd}_on_real"
        fits[smnp_name] = _fit_real_model(X_train, y_train, X_test, y_test, j, p, output_dir=results_dir,
                                          fit_name=smnp_name,
                                          include_skewness=True, g_dist=gd, n_samples=n_samples, burn_in=burn_in,
                                          n_chains=n_chains, thin=thin, store_g=True, n_cores=n_cores,
                                          is_simulated_data=False)
        _save_full_result(results_dir, smnp_name, fits[smnp_name], save_g=save_g, save_draws=save_draws)

    row = _build_replicate_row(r, split_seed, fits, cell_id={"g_dists": "+".join(g_dists)})
    pd.DataFrame([row]).to_csv(os.path.join(results_dir, "replicate_metrics.csv"), index=False)

    return row


def run_real_data_one_rep(data_path, rep_index, g_dists=("halfnormal",), output_dir="results_real_data",
                          base_seed=1000, test_size=0.4, n_samples=20000, burn_in=10000, n_chains=4, thin=1,
                          n_cores=None, resume=True, save_draws=False, save_g=False, loaded_data=None):
    os.makedirs(output_dir, exist_ok=True)
    if loaded_data is None:
        loaded_data = load_real_data(data_path)

    X, y, j, p, N, feature_names = loaded_data
    split_seed = _rep_seed(base_seed, rep_index)
    run_dir = os.path.join(output_dir, f"run_{rep_index:03d}")
    rep_csv = os.path.join(run_dir, "results", "replicate_metrics.csv")

    if resume and os.path.exists(rep_csv):
        print(f"[rep {rep_index}] found existing results at {rep_csv}; nothing to do")
        return pd.read_csv(rep_csv).iloc[0].to_dict()

    print(f"\n{'=' * 72}\nREP {rep_index}  (seed={split_seed}); {run_dir}\n{'=' * 72}")
    return run_one_real_split(rep_index, run_dir, split_seed, X, y, j, p, g_dists, test_size, n_samples, burn_in,
                              n_chains, thin, n_cores=n_cores, save_draws=save_draws, save_g=save_g)


def _summarize_difference(d, metric, contrast, positive_means):
    d = pd.Series(d).dropna()
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    return {"metric": metric, "contrast": contrast, "positive_means": positive_means, "n": n,
            "mean_diff": d.mean() if n else np.nan, "sd": sd, "mcse": sd / np.sqrt(n) if n > 1 else np.nan,
            "q2.5": d.quantile(0.025) if n else np.nan, "q97.5": d.quantile(0.975) if n else np.nan,
            "P(diff>0)": float(np.mean(d > 0)) if n else np.nan}


def aggregate_real_data_repeated(output_dir, R, j, g_dists):
    rows = []
    missing = []
    for r in range(R):
        rep_csv = os.path.join(output_dir, f"run_{r:03d}", "results", "replicate_metrics.csv")
        if os.path.exists(rep_csv):
            rows.append(pd.read_csv(rep_csv).iloc[0].to_dict())
        else:
            missing.append(r)

    if missing:
        print(f"WARNING: {len(missing)}/{R} repetitions have no replicate_metrics.csv: {missing}")
    if not rows:
        raise RuntimeError(f"No completed repetitions found under {output_dir}; nothing to aggregate.")

    df = pd.DataFrame(rows)
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    df.to_csv(os.path.join(agg_dir, "all_splits.csv"), index=False)

    summary = summarize_replicates(df)
    summary.to_csv(os.path.join(agg_dir, "summary.csv"))

    print(f"\nAGGREGATE SUMMARY ({len(rows)}/{R} repetitions found, mean/sd over splits, head):")
    with pd.option_context('display.max_rows', 30, 'display.width', 160):
        print(summary.round(4).head(30))

    higher_is_better = ["accuracy", "auc", "f1_macro", "f1_weighted"]
    lower_is_better = ["log_score", "brier_score", "MAE_choices", "KS_statistic"]

    paired_outputs = {}
    recall_outputs = {}

    for gd in g_dists:
        smnp_model = f"smnp_{gd}"
        paired_rows = []

        for metric in higher_is_better:
            try:
                d = paired_difference(df, metric, smnp_model, "mnp", "real")
            except KeyError:
                continue
            paired_rows.append(
                _summarize_difference(d, metric, contrast=f"{smnp_model} - mnp", positive_means="SMNP better"))

        for metric in lower_is_better:
            try:
                d = paired_difference(df, metric, "mnp", smnp_model, "real")
            except KeyError:
                continue
            paired_rows.append(
                _summarize_difference(d, metric, contrast=f"mnp - {smnp_model}", positive_means="SMNP better"))

        paired_df = pd.DataFrame(paired_rows)
        paired_df.to_csv(os.path.join(agg_dir, f"smnp_{gd}_vs_mnp_paired_test.csv"), index=False)
        paired_outputs[gd] = paired_df

        recall_df = per_class_recall_table(df, model_a=smnp_model, model_b="mnp", data="real", J=j)
        recall_df.to_csv(os.path.join(agg_dir, f"smnp_{gd}_vs_mnp_recall_by_class.csv"), index=False)
        recall_outputs[gd] = recall_df

        print(f"\nSMNP-{gd} vs MNP, paired over held-out splits:")
        if not paired_df.empty:
            print(paired_df.round(4))

    return df, summary, paired_outputs, recall_outputs


def run_real_data_repeated(data_path, g_dists="(halfnormal, )", output_dir="results_real_data", R=50,
                           test_size=0.4, base_seed=1000, n_samples=20000, burn_in=10000, n_chains=4, thin=1,
                           n_cores=None, resume=True, save_draws=False, save_g=False):
    os.makedirs(output_dir, exist_ok=True)
    loaded_data = load_real_data(data_path)
    _, _, j, _, _, _ = loaded_data
    pd.DataFrame({"replicate": np.arange(R), "split_seed": [_rep_seed(base_seed, r) for r in range(R)]}).to_csv(
        os.path.join(output_dir, "seed_manifest.csv"), index=False)

    for r in range(R):
        run_real_data_one_rep(data_path=data_path, rep_index=r, g_dists=g_dists, output_dir=output_dir,
                              base_seed=base_seed, test_size=test_size, n_samples=n_samples, burn_in=burn_in,
                              n_chains=n_chains, thin=thin, n_cores=n_cores, resume=resume, save_draws=save_draws,
                              save_g=save_g, loaded_data=loaded_data)

    return aggregate_real_data_repeated(output_dir, R=R, j=j, g_dists=g_dists)


def _summarize_scalar_draws(draws):
    """Posterior summary for one scalar parameter."""
    a = np.asarray(draws, dtype=float).reshape(-1)
    return {"mean": float(np.mean(a)), "median": float(np.median(a)), "sd": float(np.std(a, ddof=1)),
            "q025": float(np.percentile(a, 2.5)), "q975": float(np.percentile(a, 97.5)),
            "P_gt_0": float(np.mean(a > 0.0))}


def _stack_chain_draws(result, key):
    chains = result.get("samples_list")
    if not chains:
        raise RuntimeError("No posterior samples_list found in fitted result.")

    arrs = [np.asarray(chain[key], dtype=float) for chain in chains if key in chain]
    n_draws = [a.shape[0] for a in arrs]

    if len(set(n_draws)) != 1:
        raise ValueError(f"Chains have unequal numbers of draws for {key!r}: {n_draws}")

    return np.stack(arrs, axis=0)


def _beta_draws_standardized_to_raw(beta_draws, X_mean, X_std):
    b_draws = np.asarray(beta_draws, dtype=float)
    X_mean = np.asarray(X_mean, dtype=float)
    X_std = np.asarray(X_std, dtype=float)

    beta_copy = b_draws.copy()

    beta_copy[..., 1:] = b_draws[..., 1:] / X_std[1:]
    beta_copy[..., 0] = (b_draws[..., 0] - np.sum((b_draws[..., 1:] * X_mean[1:]) / X_std[1:], axis=-1))

    return beta_copy


def save_full_data_inference_tables(result, X, feature_names, j, g_dist, output_dir, model_label):
    os.makedirs(output_dir, exist_ok=True)
    _, _, X_mean, X_std = normalize_train_test(X, X)
    K = j - 1

    beta_std = _stack_chain_draws(result, "beta")
    beta_raw = _beta_draws_standardized_to_raw(beta_std, X_mean, X_std)

    beta_rows = []
    for a in range(K):
        alt = a + 2
        for k, predictor in enumerate(feature_names):
            s = _summarize_scalar_draws(beta_raw[:, :, a, k])
            beta_rows.append({"model": model_label, "alternative": alt, "predictor": predictor, **s})

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(os.path.join(output_dir, f"{model_label}_beta_raw_scale.csv"), index=False)

    sigma_draws = _stack_chain_draws(result, "Psi")
    sigma_rows = []
    for a in range(K):
        for b in range(a, K):
            s = _summarize_scalar_draws(sigma_draws[:, :, a, b])
            sigma_rows.append({"model": model_label, "row": a + 1, "col": b + 1,
                               "parameter": f"Sigma_{a + 1}{b + 1}", "fixed_by_identification": bool(a == 0 and b == 0),
                               **s})

    sigma_df = pd.DataFrame(sigma_rows)
    sigma_df.to_csv(os.path.join(output_dir, f"Sigma.csv"), index=False)

    alpha_df = pd.DataFrame()
    asymmetric_df = pd.DataFrame()

    if result.get("samples_list") and "delta" in result["samples_list"][0]:
        alpha_draws = _stack_chain_draws(result, "delta")
        alpha_rows = []
        for a in range(K):
            s = _summarize_scalar_draws(alpha_draws[:, :, a])
            alpha_rows.append({"model": model_label, "alternative": a + 2, "parameter": f"alpha_{a + 2}", **s})
        alpha_df = pd.DataFrame(alpha_rows)
        alpha_df.to_csv(os.path.join(output_dir, f"alpha.csv"), index=False)

        lam = lambda_from_draws(alpha_draws, sigma_draws, g_dist)
        lam_sq = lam ** 2
        gamma1 = gamma1_from_draws(alpha_draws, sigma_draws, g_dist)

        asymmetric_rows = []
        for a in range(K):
            lam_s = _summarize_scalar_draws(lam[:, :, a])
            lam2_s = _summarize_scalar_draws(lam_sq[:, :, a])
            g1_s = _summarize_scalar_draws(gamma1[:, :, a])

            asymmetric_rows.append({"model": model_label, "alternative": a + 2, "lambda_mean": lam_s["mean"],
                                    "lambda_median": lam_s["median"], "lambda_sd": lam_s["sd"],
                                    "lambda_q025": lam_s["q025"],
                                    "lambda_q975": lam_s["q975"], "lambda_sq_mean": lam2_s["mean"],
                                    "lambda_sq_median": lam2_s["median"],
                                    "lambda_sq_q025": lam2_s["q025"], "lambda_sq_q975": lam2_s["q975"],
                                    "gamma1_mean": g1_s["mean"],
                                    "gamma1_median": g1_s["median"], "gamma1_sd": g1_s["sd"],
                                    "gamma1_q025": g1_s["q025"],
                                    "gamma1_q975": g1_s["q975"], "P_gamma1_gt_0": g1_s["P_gt_0"]})

        asymmetric_df = pd.DataFrame(asymmetric_rows)
        asymmetric_df.to_csv(os.path.join(output_dir, f"asymmetry_summary.csv"), index=False)

    return {"beta": beta_df, "alpha": alpha_df, "sigma": sigma_df, "asymmetry": asymmetric_df}


def run_full_data_inference_real(data_path, g_dists="(halfnormal, )", output_dir="results_real_data", n_samples=20000,
                                 burn_in=10000, n_chains=4, thin=1, n_cores=None, fit_mnp=True, save_draws=True):
    print("\n" + "=" * 80)
    print("FULL REAL-DATA INFERENCE")
    print("=" * 80)

    X, y, j, p, N, feature_names = load_real_data(data_path)

    inference_root = os.path.join(output_dir, "inference")
    os.makedirs(inference_root, exist_ok=True)

    results = {}
    tables = {}

    if fit_mnp:
        mnp_dir = os.path.join(inference_root, "mnp")
        os.makedirs(mnp_dir, exist_ok=True)
        print(f"\nFitting MNP to all N={N} observations")

        mnp_result = _fit_real_model(X, y, X, y, j, p, output_dir=mnp_dir, fit_name="diagnostics",
                                     include_skewness=False,
                                     g_dist="halfnormal", n_samples=n_samples, burn_in=burn_in, n_chains=n_chains,
                                     thin=thin, store_g=False, n_cores=n_cores, is_simulated_data=False)
        _save_full_result(mnp_dir, "mnp_full_data", mnp_result, save_g=False, save_draws=save_draws)

        results["mnp"] = mnp_result
        tables["mnp"] = save_full_data_inference_tables(mnp_result, X, feature_names, j, "halfnormal", mnp_dir, "mnp")

    for gd in g_dists:
        model_key = f"smnp_{gd}"
        model_dir = os.path.join(inference_root, model_key)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\nFitting SMNP-{gd} to all N={N} observations")

        smnp_result = _fit_real_model(X, y, X, y, j, p, output_dir=model_dir, fit_name="diagnostics",
                                      include_skewness=True,
                                      g_dist=gd, n_samples=n_samples, burn_in=burn_in, n_chains=n_chains, thin=thin,
                                      n_cores=n_cores, is_simulated_data=False, store_g=True)
        _save_full_result(model_dir, f"{model_key}_full_data", smnp_result, save_g=False, save_draws=save_draws)

        results[model_key] = smnp_result
        tables[model_key] = save_full_data_inference_tables(smnp_result, X, feature_names, j, gd, model_dir, model_key)

        if fit_mnp:
            b_smnp = \
            tables[model_key]["beta"].rename(columns={"mean": "smnp_mean", "q025": "smnp_q025", "q975": "smnp_q975"})[
                ["alternative", "predictor", "smnp_mean", "smnp_q025", "smnp_q975"]]
            b_mnp = tables["mnp"]["beta"].rename(columns={"mean": "mnp_mean", "q025": "mnp_q025", "q975": "mnp_q975"})[
                ["alternative", "predictor", "mnp_mean", "mnp_q025", "mnp_q975"]]

            beta_compare = b_smnp.merge(b_mnp, on=["alternative", "predictor"], how="inner")
            beta_compare["mean_difference_smnp_minus_mnp"] = (beta_compare["smnp_mean"] - beta_compare["mnp_mean"])
            beta_compare.to_csv(os.path.join(model_dir, "beta_vs_mnp_full_data.csv"), index=False)

    print("\nFULL-DATA INFERENCE COMPLETE")
    print(f"Inference tables saved in: {inference_root}")

    return {"results": results, "tables": tables, "inference_dir": inference_root}


def build_real_cells(R):
    return list(range(R))


def main():
    ap = argparse.ArgumentParser(description="Real-data SMNP/MNP prediction and full-data inference.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, help="tab-separated cohort file")
    ap.add_argument("--g-dist", choices=["halfnormal", "exponential", "both"],
                    default="halfnormal", help=("SMNP mixing dists. With 'both', each prediction split fits MNP "
                                                "once, SMNP-halfnormal once, and SMNP-exponential once."))
    ap.add_argument("--out-dir", default="results_real_data")
    ap.add_argument("--test-size", type=float, default=0.4)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=10000)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--n-cores", type=int, default=None, help="Number of worker processes")
    ap.add_argument("--n-reps", type=int, default=1, help="Number of independent stratified train/test splits.")
    ap.add_argument("--base-seed", type=int, default=1000,help="Base seed for each repetition.")
    ap.add_argument("--no-resume", action="store_true", help="Refit prediction repetitions")
    ap.add_argument("--save-prediction-draws", action="store_true", help="Save chain draw files")
    ap.add_argument("--save-g", action="store_true", help="save per-observation g draws.")
    ap.add_argument("--cell-index", type=int, default=None, help="slurm array")
    ap.add_argument("--aggregate-only", action="store_true", help="Aggregate")
    ap.add_argument("--inference-only", action="store_true", help="Fit the full dataset for posterior inference.")
    ap.add_argument("--with-inference", action="store_true", help="run the full-data posterior inference fits.")
    ap.add_argument("--smnp-only-inference", action="store_true", help="Skip MNP during full-data inference")
    args = ap.parse_args()

    if args.n_reps < 1:
        raise SystemExit("--n-reps must be at least 1")

    os.makedirs(args.out_dir, exist_ok=True)
    g_dists = _resolve_g_dists(args.g_dist)
    pred_root = _prediction_root(args.out_dir, g_dists)

    print("REAL DATA ANALYSIS: SKEWED MULTINOMIAL PROBIT vs MULTINOMIAL PROBIT")
    print("=" * 70)
    print(f"data : {args.data_path}")
    print(f"g_dist(s)   : {', '.join(g_dists)}")
    print(f"output : {args.out_dir}")

    if args.inference_only:
        run_full_data_inference_real(data_path=args.data_path, g_dists=g_dists, output_dir=args.out_dir,
                                     n_samples=args.n_samples, burn_in=args.burn_in, n_chains=args.n_chains,
                                     thin=args.thin, n_cores=args.n_cores, fit_mnp=not args.smnp_only_inference, save_draws=True)
        return

    if args.with_inference and args.cell_index is not None:
        raise SystemExit("--with-inference must not be used with --cell-index, because every "
                         "array task would redundantly refit the full dataset. Run full-data "
                         "inference once separately with --inference-only.")

    if args.aggregate_only:
        _, _, j, _, _, _ = load_real_data(args.data_path)
        aggregate_real_data_repeated(pred_root, R=args.n_reps, j=j, g_dists=g_dists)
        print(f"\nAggregate output: {pred_root}/aggregate/")
        return

    if args.cell_index is not None:
        cells = build_real_cells(args.n_reps)
        if args.cell_index not in cells:
            raise SystemExit(f"--cell-index {args.cell_index} out of range 0..{len(cells) - 1}")
        rep_index = args.cell_index
        print(f"[cell {args.cell_index}] repetition {rep_index}/{args.n_reps - 1}")
        print(f"[cell {args.cell_index}] output - {pred_root}")

        run_real_data_one_rep(
            data_path=args.data_path, rep_index=rep_index, g_dists=g_dists, output_dir=pred_root,
            base_seed=args.base_seed, test_size=args.test_size, n_samples=args.n_samples, burn_in=args.burn_in,
            n_chains=args.n_chains, thin=args.thin, n_cores=args.n_cores, resume=not args.no_resume,
            save_draws=args.save_prediction_draws, save_g=args.save_g)

        print(f"\n[cell {args.cell_index}] DONE. Run --aggregate-only after all repetitions finish.")
        return

    print(f"prediction repetitions: {args.n_reps} (base_seed={args.base_seed})")
    run_real_data_repeated(
        data_path=args.data_path, g_dists=g_dists, output_dir=pred_root, R=args.n_reps,
        test_size=args.test_size, base_seed=args.base_seed, n_samples=args.n_samples, burn_in=args.burn_in,
        n_chains=args.n_chains, thin=args.thin, n_cores=args.n_cores, resume=not args.no_resume,
        save_draws=args.save_prediction_draws, save_g=args.save_g)


    print("\n" + "=" * 70)
    print("PREDICTIVE ANALYSIS COMPLETE")
    print(f"Per-split results : {pred_root}/run_NNN/")
    print(f"Aggregate results : {pred_root}/aggregate/")
    print("=" * 70)

    if args.with_inference:
        run_full_data_inference_real(
            data_path=args.data_path,
            g_dists=g_dists,
            output_dir=args.out_dir,
            n_samples=args.n_samples,
            burn_in=args.burn_in,
            n_chains=args.n_chains,
            thin=args.thin,
            n_cores=args.n_cores,
            fit_mnp=not args.smnp_only_inference,
            save_draws=True,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
