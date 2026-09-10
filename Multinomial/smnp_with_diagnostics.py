import pandas as pd
import multiprocessing as mp
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
import os

from smnp_implementation import SMNPGibbsSampler
from smnp_diagnostics import SMNPDiagnostics, _az_from_posterior


def _delta_recovery_summary(delta_draws, delta_true, constrain_skewness):
    delta_draws = np.asarray(delta_draws, dtype=float)
    truth = np.asarray(delta_true, dtype=float)

    delta_mean = np.mean(delta_draws, axis=0)
    delta_median = np.median(delta_draws, axis=0)

    delta_hybrid = delta_mean.copy()
    if constrain_skewness:
        delta_hybrid[0] = delta_median[0]

    estimates = {'mean': delta_mean, 'median': delta_median, 'hybrid': delta_hybrid}
    rmses = {k: float(np.sqrt(np.mean((truth - v) ** 2))) for k, v in estimates.items()}
    return estimates, rmses


def _delta_posterior_probabilities(delta_draws, epsilon=0.1):
    delta_draws = np.asarray(delta_draws, dtype=float)
    prob_positive = np.mean(delta_draws > 0.0, axis=0)
    prob_nontrivial = np.mean(np.abs(delta_draws) > epsilon, axis=0)
    median = np.median(delta_draws, axis=0)
    return prob_positive, prob_nontrivial, median


def _delta_rhat(samples_list):
    try:
        import arviz as az
    except Exception:
        return None
    chains = [s['delta'] for s in samples_list if 'delta' in s]
    if len(chains) < 2:
        return None
    m = min(c.shape[0] for c in chains)
    stacked = np.stack([c[:m] for c in chains], axis=0)  # (chains, draws, J-1)
    idata = _az_from_posterior({'delta': stacked})
    rhat = np.asarray(az.rhat(idata)['delta'].values)
    return np.atleast_1d(rhat)


def normalize_train_test(X_train, X_test):
    train_is_df = isinstance(X_train, pd.DataFrame)

    if train_is_df:
        columns = X_train.columns
        train_index = X_train.index
        test_index = X_test.index

        X_train_array = X_train.to_numpy(dtype=float)
        X_test_array = X_test.to_numpy(dtype=float)
    else:
        columns = None
        train_index = None
        test_index = None

        X_train_array = np.asarray(X_train, dtype=float)
        X_test_array = np.asarray(X_test, dtype=float)

    p = X_train_array.shape[1]

    X_mean = np.zeros(p)
    X_std = np.ones(p)

    for k in range(p):
        col = X_train_array[:, k]

        is_constant = np.std(col) < 1e-12
        is_binary = set(np.unique(col)).issubset({0.0, 1.0})

        if not is_constant and not is_binary:
            X_mean[k] = np.mean(col)
            X_std[k] = np.std(col)

            if X_std[k] < 1e-12:
                X_std[k] = 1.0

    X_train_norm = (X_train_array - X_mean) / X_std
    X_test_norm = (X_test_array - X_mean) / X_std

    if train_is_df:
        X_train_norm = pd.DataFrame(X_train_norm, columns=columns, index=train_index)
        X_test_norm = pd.DataFrame(X_test_norm, columns=columns, index=test_index)

    return X_train_norm, X_test_norm, X_mean, X_std


def _try_plot(label, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"  [warn] {label} skipped ({type(e).__name__}: {e})")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


def run_single_chain(chain_id, y_train, X_train, j, p, include_skewness, n_samples, burn_in, thin,
                     base_seed=42, g_dist="halfnormal", constrain_skewness=False,
                     store_g=True):
    chain_seed = base_seed + (chain_id * 10)
    np.random.seed(chain_seed)

    print(f" Starting Chain {chain_id + 1} (PID: {mp.current_process().pid}, Seed: {chain_seed})")

    start_time = time.time()

    sampler = SMNPGibbsSampler(y_train, X_train, j, p, include_skewness=include_skewness, g_dist=g_dist,
                               constrain_skewness=constrain_skewness)
    samples = sampler.sample(n_samples=n_samples, burn_in=burn_in, thin=thin,
                             verbose=False, store_g=store_g)

    bad = {k: np.asarray(v).dtype for k, v in samples.items()
           if np.asarray(v).dtype != np.float64}
    if bad:
        print(f"  [dtype] chain {chain_id}: non-float arrays {bad}")

    end_time = time.time()
    print(f" Chain {chain_id + 1} completed in {end_time - start_time:.1f} seconds")

    return chain_id, samples


def _allocated_cores():
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, mp.cpu_count())


def run_parallel_chains(y_train, X_train, j, p, include_skewness, n_chains=4, n_samples=20000, burn_in=10000,
                        n_cores=None, thin=5, base_seed=42, g_dist="halfnormal", constrain_skewness=False,
                        store_g=True):
    if n_cores is None:
        n_cores = _allocated_cores()
    n_cores = min(n_cores, n_chains)
    print(f"Running {n_chains} MCMC chains in parallel using {n_cores} core(s) "
          f"(allocated={_allocated_cores()}, node={mp.cpu_count()})")

    chain_runner = partial(
        run_single_chain,
        y_train=y_train,
        X_train=X_train,
        j=j,
        p=p,
        include_skewness=include_skewness,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        base_seed=base_seed,
        g_dist=g_dist,
        constrain_skewness=constrain_skewness,
        store_g=store_g
    )

    start_time = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_cores, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(chain_runner, range(n_chains), chunksize=1))

    end_time = time.time()
    print(f"All chains completed in {end_time - start_time:.1f} seconds")

    results.sort(key=lambda x: x[0])

    # samplers = [result[1] for result in results]
    samples_list = [result[1] for result in results]

    return samples_list


def run_smnp_with_diagnostics(X_train, y_train, X_test, y_test, j, p, beta_true=None, delta_true=None, true_params=None,
                              experiment_name="SMNP Experiment", is_simulated_data=None, output_dir="results",
                              include_skewness=True, n_cores=None, g_dist="halfnormal", constrain_skewness=False,
                              n_samples=20000, burn_in=10000, n_chains=4, thin=1,
                              store_g=True):
    """
    Complete workflow for SMNP modeling with parallel chain execution
    """
    if g_dist not in {"halfnormal", "exponential"}:
        raise ValueError("g_dist must be either 'halfnormal' or 'exponential'.")

    save_path = os.path.join(output_dir, experiment_name.replace(" ", "_").lower())
    os.makedirs(save_path, exist_ok=True)

    if is_simulated_data is None:
        is_simulated_data = (beta_true is not None)  # and (delta_true is not None)
        detection_method = "auto-detected"
    else:
        detection_method = "explicitly set"

    has_true_params = (beta_true is not None)  # and (delta_true is not None)
    if is_simulated_data and not has_true_params:
        print(" WARNING: is_simulated_data=True but no true parameters provided!")
        print(" Setting is_simulated_data=False (real data mode)")
        is_simulated_data = False
    elif not is_simulated_data and has_true_params:
        print("INFO: is_simulated_data=False but true parameters provided")
        print("True parameters will be ignored (real data mode)")

    print("=" * 80)
    print("MODEL COMPREHENSIVE ANALYSIS")
    print(f"DATA TYPE: {'SIMULATED' if is_simulated_data else 'REAL'} ({detection_method})")
    if is_simulated_data:
        print("Parameter recovery metrics will be computed")
    else:
        print("Parameter estimates only (no recovery metrics)")
    print("=" * 80)

    X_train_used, X_test_used, X_mean, X_std = normalize_train_test(X_train, X_test)

    print("\n FITTING MODEL")
    print("-" * 40)

    start_time = time.time()
    samples_list = run_parallel_chains(y_train=y_train, X_train=X_train_used, j=j, p=p,
                                       include_skewness=include_skewness,
                                       n_chains=n_chains, n_samples=n_samples, burn_in=burn_in, thin=thin,
                                       n_cores=n_cores, base_seed=42, g_dist=g_dist,
                                       constrain_skewness=constrain_skewness,
                                       store_g=store_g)

    total_time = time.time() - start_time
    print(f"All chains completed successfully in {total_time:.1f} seconds!")

    template_sampler = SMNPGibbsSampler(y_train, X_train_used, j, p, include_skewness=include_skewness, g_dist=g_dist,
                                        constrain_skewness=constrain_skewness)

    print("\n COMPREHENSIVE DIAGNOSTICS")
    print("-" * 40)

    # Conditional default variables based on model type
    if include_skewness:
        default_vars = ['beta', 'delta']
    else:
        default_vars = ['beta']

    if j - 2 > 0:
        default_vars.append('gamma')

    diagnostics = SMNPDiagnostics(template_sampler, samples_list, y_train, X_test=X_test_used, y_test=y_test,
                                  default_var_names=default_vars)

    print("\nPOSTERIOR SUMMARY:")
    summary = diagnostics.summarize()
    summary.to_csv(os.path.join(save_path, "posterior_summary.csv"), index=False)

    print("\nCONVERGENCE ASSESSMENT:")
    conv_diag = diagnostics.convergence_diagnostics()
    conv_diag.to_csv(os.path.join(save_path, "convergence_diagnostics.csv"), index=False)

    def beta_standardized_to_raw(beta_std, X_mean, X_std, j, p):
        """
        Convert beta estimated using normalized X back to raw-X scale.
        """
        beta_std = np.asarray(beta_std, dtype=float).reshape(j - 1, p)
        X_mean = np.asarray(X_mean, dtype=float)
        X_std = np.asarray(X_std, dtype=float)

        beta_raw = beta_std.copy()
        beta_raw[:, 1:] = beta_std[:, 1:] / X_std[1:]
        beta_raw[:, 0] = beta_std[:, 0] - np.sum(beta_std[:, 1:] * X_mean[1:] / X_std[1:], axis=1)

        return beta_raw.reshape(-1)

    beta_rmse = None
    delta_rmse = None
    delta_rmse_mean = None
    delta_rmse_median = None
    delta_rmse_hybrid = None
    derived = None

    if is_simulated_data:
        print("\n PARAMETER RECOVERY ASSESSMENT")
        print("-" * 40)

        # Extract posterior means using combined samples
        combined = diagnostics._combine_chains(preserve_chains=False)  # Get flattened samples
        beta_est_std = np.mean(combined['beta'], axis=0)
        beta_est = beta_standardized_to_raw(beta_est_std, X_mean, X_std, j, p)

        beta_rmse = np.sqrt(np.mean((beta_true - beta_est) ** 2))

        delta_estimates = None

        if include_skewness:
            constrain = getattr(template_sampler, 'constrain_skewness', False)
            delta_estimates, delta_rmses = _delta_recovery_summary(
                combined['delta'], delta_true, constrain)
            delta_rmse_mean = delta_rmses['mean']
            delta_rmse_median = delta_rmses['median']
            delta_rmse_hybrid = delta_rmses['hybrid']
            delta_rmse = delta_rmse_hybrid

        print("COEFFICIENT PARAMETERS (beta):")
        for i, (true_val, est_val) in enumerate(zip(beta_true, beta_est)):
            error = abs(true_val - est_val)
            print(f" beta_{i}: {true_val:+.3f} → {est_val:+.3f} (error: {error:.3f})")

        if include_skewness:
            d_mean = delta_estimates['mean']
            d_med = delta_estimates['median']
            print("\nSKEWNESS PARAMETERS (delta):  true -> [mean | median]")
            for i, true_val in enumerate(delta_true):
                alt_idx = i + 2
                print(f" delta_{alt_idx}: {true_val:+.3f} -> "
                      f"[{d_mean[i]:+.3f} | {d_med[i]:+.3f}]")

        print(f"beta RMSE: {beta_rmse:.4f}")
        if include_skewness:
            print(f"delta RMSE  (mean)   : {delta_rmse_mean:.4f}")
            print(f"delta RMSE  (median) : {delta_rmse_median:.4f}")
            print(f"delta RMSE  (hybrid) : {delta_rmse_hybrid:.4f}")

        if 'Psi' in combined:
            Psi_draws = combined['Psi']  # (n_draws, J-1, J-1)
            diag_draws = np.diagonal(Psi_draws, axis1=1, axis2=2)  # (n_draws, J-1)
            sig_mean = diag_draws.mean(axis=0)
            sig_lo = np.percentile(diag_draws, 2.5, axis=0)
            sig_hi = np.percentile(diag_draws, 97.5, axis=0)
            Psi_true_mat = true_params.get('Psi') if true_params else None
            print("\nSIGMA DIAGONAL (ridge diagnostic; Sigma_11 fixed = 1):")
            sigma_rows = []
            for jj in range(sig_mean.shape[0]):
                t = float(Psi_true_mat[jj, jj]) if Psi_true_mat is not None else np.nan
                print(f" Sigma_{jj + 1}{jj + 1}: true {t:6.3f} → {sig_mean[jj]:6.3f} "
                      f"[{sig_lo[jj]:.3f}, {sig_hi[jj]:.3f}]")
                sigma_rows.append({'param': f'Sigma_{jj + 1}{jj + 1}', 'true': t,
                                   'mean': sig_mean[jj], '2.5%': sig_lo[jj],
                                   '97.5%': sig_hi[jj]})
            pd.DataFrame(sigma_rows).to_csv(
                os.path.join(save_path, 'sigma_summary.csv'), index=False)

        if include_skewness and 'delta' in combined and 'Psi' in combined:
            from derived_quantities import derived_summary
            derived = derived_summary(combined['delta'], combined['Psi'], g_dist,
                                      true_delta=delta_true)
            print("\nDERIVED (IDENTIFIED) QUANTITIES  --  report these, not raw delta/Sigma:")
            lam_lo = derived['lambda_star_sq_q025'] ** 0.5
            lam_hi = derived['lambda_star_sq_q975'] ** 0.5
            print(f" lambda_*  : {derived['lambda_star_mean']:.3f}  "
                  f"[{lam_lo:.3f}, {lam_hi:.3f}]")
            for jj in range(len(delta_true)):
                pk = f"j{jj + 1}"
                print(f" |lambda_{jj + 2}| : {derived[f'abs_lambda_{pk}_mean']:.3f}   "
                      f"Omega_{jj + 2}{jj + 2} : {derived[f'omega_{pk}_mean']:.3f}")
            pd.DataFrame([derived]).to_csv(
                os.path.join(save_path, 'derived_quantities_summary.csv'), index=False)

    else:
        print("\n PARAMETER ESTIMATES (REAL DATA)")
        print("-" * 40)

        combined = diagnostics._combine_chains(preserve_chains=False)
        beta_est_std = np.mean(combined['beta'], axis=0)
        beta_est = beta_standardized_to_raw(beta_est_std, X_mean, X_std, j, p)

        print("ESTIMATED COEFFICIENT PARAMETERS (beta, raw covariate scale):")

        for i, est_val in enumerate(beta_est):
            print(f" beta_{i}: {est_val:+.3f}")

        if include_skewness and 'delta' in combined:
            prob_pos, prob_nt, d_med = _delta_posterior_probabilities(
                combined['delta'], epsilon=0.1)
            rhat = _delta_rhat(samples_list)
            print("\nESTIMATED SKEWNESS PARAMETERS (delta):")
            print("  posterior median, P(delta>0 | data), P(|delta|>0.1 | data)")
            if rhat is not None and np.nanmax(rhat) > 1.05:
                print(f"max delta R-hat = {np.nanmax(rhat):.3f} > 1.05.")
            for i in range(len(d_med)):
                alt_idx = i + 2
                print(f" delta_{alt_idx}: median={d_med[i]:+.3f}  "
                      f"P(>0)={prob_pos[i]:.3f}  P(|.|>0.1)={prob_nt[i]:.3f}")

        if include_skewness and 'delta' in combined and 'Psi' in combined:
            from derived_quantities import derived_summary
            derived = derived_summary(combined['delta'], combined['Psi'], g_dist)
            print("\nDERIVED (IDENTIFIED) QUANTITIES  --  report these, not raw delta/Sigma:")
            lam_lo = derived['lambda_star_sq_q025'] ** 0.5
            lam_hi = derived['lambda_star_sq_q975'] ** 0.5
            print(f" lambda_*  : {derived['lambda_star_mean']:.3f}  "
                  f"[{lam_lo:.3f}, {lam_hi:.3f}]")
            pd.DataFrame([derived]).to_csv(
                os.path.join(save_path, 'derived_quantities_summary.csv'), index=False)

    print("\n EVALUATING MODEL PERFORMANCE")
    print("-" * 40)

    metric_names = ["accuracy", "auc", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "log_score",
                    "brier_score", "MAE_choices", "KS_statistic", "DIC", "pD", 'p_DIC']

    def save_metric_dict(metrics, filepath):
        with open(filepath, "w") as f:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

    print("Evaluating training conditional fit...")
    train_conditional_metrics = diagnostics.evaluate_train_performance(X_train=X_train_used, y_train=y_train,
                                                                       samples_list=samples_list,
                                                                       prediction_mode="conditional",
                                                                       label="train_conditional", n_samples=300,
                                                                       n_mc=50)

    print("Evaluating training marginal prediction...")
    train_marginal_metrics = diagnostics.evaluate_train_performance(X_train=X_train_used, y_train=y_train,
                                                                    samples_list=samples_list,
                                                                    prediction_mode="marginal",
                                                                    label="train_marginal", n_samples=300, n_mc=50)

    print("Evaluating test marginal prediction...")
    test_marginal_metrics = diagnostics.evaluate_test_performance(X_test=X_test_used, y_test=y_test,
                                                                  prediction_mode="marginal", label="test_marginal",
                                                                  n_samples=300, n_mc=50)

    metrics_by_label = {"train_conditional": train_conditional_metrics, "train_marginal": train_marginal_metrics,
                        "test_marginal": test_marginal_metrics}

    for label, metrics in metrics_by_label.items():
        save_metric_dict(metrics, os.path.join(save_path, f"{label}_performance_metrics.txt"))

    rows = []

    for label, metrics in metrics_by_label.items():
        row = {"evaluation_label": label, "data_type": metrics.get("data_type"),
               "prediction_mode": metrics.get("prediction_mode")}

        for metric in metric_names:
            row[metric] = metrics.get(metric, np.nan)

        rows.append(row)

    prediction_metrics_df = pd.DataFrame(rows)
    prediction_metrics_df.to_csv(os.path.join(save_path, "prediction_metrics_by_mode.csv"), index=False)

    comparison_rows = []
    for metric in ["accuracy", "auc", "f1_macro", "log_score", "brier_score"]:
        train_cond = train_conditional_metrics.get(metric, np.nan)
        train_marg = train_marginal_metrics.get(metric, np.nan)
        test_marg = test_marginal_metrics.get(metric, np.nan)
        comparison_rows.append({
            "Metric": metric,
            "Train conditional": train_cond,
            "Train marginal": train_marg,
            "Test marginal": test_marg,
            "Test marginal - Train marginal": test_marg - train_marg,
            "Train conditional - Train marginal": train_cond - train_marg
        })

    prediction_comparison_df = pd.DataFrame(comparison_rows)
    prediction_comparison_df.to_csv(os.path.join(save_path, "prediction_mode_comparison.csv"), index=False)

    performance_comparison = pd.DataFrame([
        {"Metric": metric, "Training": train_marginal_metrics.get(metric, np.nan),
         "Test": test_marginal_metrics.get(metric, np.nan)}
        for metric in ["accuracy", "auc", "f1_macro", "log_score", "brier_score"]])
    performance_comparison.to_csv(os.path.join(save_path, "train_test_comparison.csv"), index=False)

    train_metrics = train_marginal_metrics
    test_metrics = test_marginal_metrics

    with open(os.path.join(save_path, "classification_report_train_conditional.txt"), "w") as f:
        f.write(
            diagnostics.classification_report(X=X_train_used, y=y_train, data='train', prediction_mode="conditional",
                                              use_mean_params=False, n_samples=300, n_mc=50))

    with open(os.path.join(save_path, "classification_report_train_marginal.txt"), "w") as f:
        f.write(diagnostics.classification_report(X=X_train_used, y=y_train, data='train', prediction_mode="marginal",
                                                  use_mean_params=False, n_samples=300, n_mc=50))

    with open(os.path.join(save_path, "classification_report_test_marginal.txt"), "w") as f:
        f.write(diagnostics.classification_report(X=X_test_used, y=y_test, data='test', prediction_mode="marginal",
                                                  use_mean_params=False, n_samples=300, n_mc=50))

    print("\n6. GENERATING DIAGNOSTIC PLOTS")
    print("-" * 40)

    plt.style.use('default')
    print("Creating trace plots...")

    def _traces():
        diagnostics.trace_plots(figsize=(16, 24))
        plt.savefig(os.path.join(save_path, "trace_plots.png"))

    _try_plot("trace plots", _traces)

    print("Creating autocorrelation plots...")
    _try_plot("autocorrelation plots", diagnostics.autocorrelation_plots,
              max_lag=100, figsize=(12, 8),
              save_path=os.path.join(save_path, "autocorrelation_plots.png"))

    true_values_dict = None
    if is_simulated_data and beta_true is not None:
        true_values_dict = {}
        beta_matrix_raw = np.asarray(beta_true, dtype=float).reshape((j - 1, p))
        beta_matrix = beta_matrix_raw * np.asarray(X_std)[None, :]
        beta_matrix[:, 0] = beta_matrix_raw[:, 0] + np.sum(beta_matrix_raw[:, 1:] * np.asarray(X_mean)[None, 1:],
                                                           axis=1)

        for alt_idx in range(j - 1):
            for cov_idx in range(p):
                name = f'beta_{alt_idx + 2}_{cov_idx}'
                true_values_dict[name] = float(beta_matrix[alt_idx, cov_idx])

        if include_skewness and delta_true is not None:
            for alt_idx, dval in enumerate(delta_true):
                true_values_dict[f'delta_{alt_idx + 2}'] = float(dval)

        if j - 2 > 0 and true_params is not None and 'gamma' in true_params:
            for gidx, gval in enumerate(true_params['gamma']):
                true_values_dict[f'gamma_2_{gidx + 2}'] = float(gval)

    print("Creating posterior plots...")
    _try_plot("posterior plots", diagnostics.posterior_plots,
              figsize=(18, 24),
              save_path=os.path.join(save_path, "posterior_plots.png"),
              true_values=true_values_dict)

    print("Creating training conditional confusion matrix...")
    diagnostics.confusion_matrix_plot(X=X_train_used, y=y_train, figsize=(8, 6), data='train',
                                      prediction_mode="conditional", use_mean_params=False, n_samples=300, n_mc=50)
    plt.savefig(os.path.join(save_path, "train_conditional_confusion_matrix.png"))
    plt.close()

    print("Creating training marginal confusion matrix...")
    diagnostics.confusion_matrix_plot(X=X_train_used, y=y_train, figsize=(8, 6), data='train',
                                      prediction_mode="marginal", use_mean_params=False, n_samples=300, n_mc=50)
    plt.savefig(os.path.join(save_path, "train_marginal_confusion_matrix.png"))
    plt.close()

    print("Creating test marginal confusion matrix...")
    diagnostics.confusion_matrix_plot(X=X_test_used, y=y_test, figsize=(8, 6), data='test', prediction_mode="marginal",
                                      use_mean_params=False, n_samples=300, n_mc=50)
    plt.savefig(os.path.join(save_path, "test_marginal_confusion_matrix.png"))
    plt.close()

    print("Creating performance comparison plot...")
    _plot_train_test_comparison(performance_comparison, save_path)

    print("Creating calibration plot...")
    _plot_prediction_calibration(diagnostics, X_train_used, y_train, X_test_used, y_test, save_path)

    print("\n7. ANALYSIS SUMMARY")
    print("-" * 40)

    if is_simulated_data:
        if beta_rmse < 0.2:
            print(" Beta Recovery: EXCELLENT")
        elif beta_rmse < 0.5:
            print(" Beta Recovery: GOOD")
        else:
            print(" Beta Recovery: MODERATE")

        if include_skewness and derived is not None:
            ls_lo = derived['lambda_star_sq_q025'] ** 0.5
            ls_hi = derived['lambda_star_sq_q975'] ** 0.5
            true_ls = None
            if true_params is not None and 'delta' in true_params and 'Psi' in true_params:
                from derived_quantities import lambda_star_sq
                true_ls = float(np.sqrt(lambda_star_sq(
                    np.asarray(delta_true)[None, None, :],
                    np.asarray(true_params['Psi'])[None, None, :, :], g_dist))[0, 0])
            if true_ls is not None:
                covers = ls_lo <= true_ls <= ls_hi
                print(f" Sigma*/lambda_* Recovery: lambda_* true={true_ls:.3f}, "
                      f"95% CI=[{ls_lo:.3f}, {ls_hi:.3f}] "
                      f"({'COVERS' if covers else 'DOES NOT COVER'} truth)")
            else:
                print(f" lambda_* posterior: {derived['lambda_star_mean']:.3f} "
                      f"[{ls_lo:.3f}, {ls_hi:.3f}] (no truth available to check coverage)")
            print(f" delta_rmse (hybrid) = {delta_rmse:.3f} -- reported for reference only")
    else:
        print(" Parameter Recovery not available for real data")

    train_accuracy = train_marginal_metrics.get('accuracy', np.nan)
    test_accuracy = test_marginal_metrics.get('accuracy', np.nan)
    accuracy_drop = train_accuracy - test_accuracy

    if np.isfinite(accuracy_drop):
        if accuracy_drop <= 0.05:
            print(" Generalization: GOOD")
        else:
            print(" Generalization: POOR (possible overfitting)")
    else:
        print(" Generalization: unavailable")

    if np.isfinite(test_accuracy):
        if test_accuracy > 0.7:
            print(" Test Accuracy: GOOD")
        else:
            print(" Test Accuracy: MODERATE")

    if include_skewness:
        print("\nSkewness Effects Detected (posterior evidence):")
        combined = diagnostics._combine_chains(preserve_chains=False)
        if 'delta' in combined:
            prob_pos, prob_nt, d_med = _delta_posterior_probabilities(
                combined['delta'], epsilon=0.1)
            rhat = _delta_rhat(samples_list)
            if rhat is not None and np.nanmax(rhat) > 1.05:
                print(f"  [warning] max delta R-hat = {np.nanmax(rhat):.3f} > 1.05: "
                      "reflection may be unresolved; directional probabilities unreliable.")
            alt_names = [f"Alt {jj}" for jj in range(2, j + 1)]
            for i, name in enumerate(alt_names):
                direction = "positive" if prob_pos[i] >= 0.5 else "negative"
                p_dir = prob_pos[i] if direction == "positive" else 1.0 - prob_pos[i]
                if prob_nt[i] >= 0.9 and p_dir >= 0.9:
                    verdict = f"strong {direction}"
                elif prob_nt[i] >= 0.5:
                    verdict = f"moderate {direction}"
                else:
                    verdict = "minimal / uncertain"
                print(f" {name}: {verdict}  "
                      f"(median={d_med[i]:+.3f}, P(delta>0)={prob_pos[i]:.3f}, "
                      f"P(|delta|>0.1)={prob_nt[i]:.3f})")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    with open(os.path.join(save_path, "computational_info.txt"), "w") as f:
        f.write(f"Total computation time: {total_time:.1f} seconds\n")
        f.write(f"Number of chains: {n_chains}\n")
        f.write(f"CPU cores used: {min(n_cores or mp.cpu_count() - 1, n_chains)}\n")
        f.write(f"Available CPU cores: {mp.cpu_count()}\n")
        f.write(f"Samples per chain: {n_samples}\n")
        f.write(f"Burn-in per chain: {burn_in}\n")
        f.write(f"Thin: {thin}\n")
        f.write(f"Skewing distribution g_dist: {g_dist}\n")
        f.write(f"Constrain skewness: {constrain_skewness}\n")
        f.write(f"Data type: {'Simulated' if is_simulated_data else 'Real'} ({detection_method})\n")

    print(f"\n Analysis complete! Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {save_path}")

    return {
        'samplers': template_sampler,
        'samples_list': samples_list,
        'diagnostics': diagnostics,
        'true_params': true_params if is_simulated_data else None,
        'train_conditional_metrics': train_conditional_metrics,
        'train_marginal_metrics': train_marginal_metrics,
        'test_marginal_metrics': test_marginal_metrics,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'prediction_metrics_by_mode': prediction_metrics_df,
        'prediction_mode_comparison': prediction_comparison_df,
        'performance_comparison': performance_comparison,
        'beta_rmse': beta_rmse,
        'delta_rmse': delta_rmse,
        'delta_rmse_mean': delta_rmse_mean,
        'delta_rmse_median': delta_rmse_median,
        'delta_rmse_hybrid': delta_rmse_hybrid,
        'derived_quantities': derived,
        'computation_time': total_time,
        'is_simulated_data': is_simulated_data,
        'save_path': save_path
    }


def _plot_parameter_comparison(beta_true, beta_est, delta_true, delta_est, save_path, include_skewness=True):
    if include_skewness and delta_est is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    ax1.scatter(beta_true, beta_est, alpha=0.7, s=60, label="Beta Parameters")
    min_val, max_val = min(beta_true.min(), beta_est.min()), max(beta_true.max(), beta_est.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Recovery')
    ax1.set_xlabel('True β Values')
    ax1.set_ylabel('Estimated β Values')
    ax1.set_title('Coefficient Recovery (β)')
    ax1.legend()
    ax1.grid(False)

    if include_skewness and delta_est is not None:
        print("Creating delta parameter comparsion plots...")
        ax2.scatter(delta_true, delta_est, alpha=0.7, s=60, label="Delta Parameters", color='orange')
        min_val, max_val = min(delta_true.min(), delta_est.min()), max(delta_true.max(), delta_est.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Recovery')
        ax2.set_xlabel('True delta Values')
        ax2.set_ylabel('Estimated delta Values')
        ax2.set_title('Skewness Recovery (delta)')
        ax2.legend()
        ax2.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "parameter_comparison_plots.png"))
    plt.close()


def _plot_train_test_comparison(performance_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = performance_df['Metric'].values
    train_vals = performance_df['Training'].values
    test_vals = performance_df['Test'].values

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_vals, width, label='Training', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test', alpha=0.8, color='lightcoral')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Training vs Test Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "train_test_comparison.png"))
    plt.close()


def _plot_prediction_calibration(diagnostics, X_train, y_train, X_test, y_test, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    train_probs = diagnostics.predict_probabilities(
        X_train, use_mean_params=False, n_samples=300, is_training_data=True,
        training_indices=np.arange(len(y_train)), n_mc=50, prediction_mode="marginal")
    train_pred_conf = np.max(train_probs, axis=1)
    train_pred_choices = np.argmax(train_probs, axis=1) + 1
    train_correct = (train_pred_choices == y_train).astype(int)

    test_probs = diagnostics.predict_probabilities(
        X_test, use_mean_params=False, n_samples=300, is_training_data=False,
        n_mc=50, prediction_mode="marginal")
    test_pred_conf = np.max(test_probs, axis=1)
    test_pred_choices = np.argmax(test_probs, axis=1) + 1
    test_correct = (test_pred_choices == y_test).astype(int)

    for i, (conf, correct, title, ax) in enumerate([(train_pred_conf, train_correct, 'Training Calibration', ax1),
                                                    (test_pred_conf, test_correct, 'Test Calibration', ax2)]):
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf > bin_lower) & (conf <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = conf[in_bin].mean()

                bin_centers.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(in_bin.sum())

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.scatter(bin_centers, bin_accuracies, s=[c * 5 for c in bin_counts], alpha=0.7, label='Actual Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "prediction_calibration.png"))
    plt.close()
