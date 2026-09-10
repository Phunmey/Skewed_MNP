import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")


def _test_col(metric, model, data):
    return f"test_metrics__{metric}__{model}_on_{data}"


def _recovery_col(kind, model, data):
    return f"{kind}__{model}_on_{data}"


def _recov_experiment_col(kind, model, data):
    return f"{kind}__{model}_on_{data}_recov"


def _derived_recov_col(field, model, data):
    return f"derived__{field}__{model}_on_{data}_recov"


def _mean_sd(df, col):
    if col not in df.columns:
        return np.nan, np.nan, 0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    sd = s.std(ddof=1) if len(s) > 1 else np.nan
    return s.mean(), sd, len(s)


def _paired_diff_mean_sd(df, metric, model_a, model_b, data):
    ca = _test_col(metric, model_a, data)
    cb = _test_col(metric, model_b, data)
    if ca not in df.columns or cb not in df.columns:
        return np.nan, np.nan, np.nan, 0
    d = (pd.to_numeric(df[ca], errors="coerce") - pd.to_numeric(df[cb], errors="coerce")).dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    p_gt0 = float(np.mean(d > 0))
    return d.mean(), sd, p_gt0, len(d)


def create_aggregate_summary(all_sample_results, save_dir="results_aggregate"):
    rows = []

    for _, cell in all_sample_results.items():
        N = cell["sample_size"]
        cell_dir = cell["output_dir"]
        rep_csv = os.path.join(cell_dir, "aggregate", "all_replicates.csv")
        if not os.path.exists(rep_csv):
            print(f"[trend] missing {rep_csv}; skipping N={N}")
            continue
        df = pd.read_csv(rep_csv)

        for data in ("smnp", "mnp"):  # skewed-data pairing, symmetric-data pairing
            acc_s_m, acc_s_sd, _ = _mean_sd(df, _test_col("accuracy", "smnp", data))
            acc_m_m, acc_m_sd, _ = _mean_sd(df, _test_col("accuracy", "mnp", data))
            acc_d_m, acc_d_sd, acc_p, n_acc = _paired_diff_mean_sd(df, "accuracy", "smnp", "mnp", data)

            f1_s_m, _, _ = _mean_sd(df, _test_col("f1_macro", "smnp", data))
            f1_m_m, _, _ = _mean_sd(df, _test_col("f1_macro", "mnp", data))
            f1_d_m, f1_d_sd, f1_p, _ = _paired_diff_mean_sd(df, "f1_macro", "smnp", "mnp", data)

            br_s_m, _, _ = _mean_sd(df, _test_col("brier_score", "smnp", data))
            br_m_m, _, _ = _mean_sd(df, _test_col("brier_score", "mnp", data))
            br_d_m, br_d_sd, br_p, _ = _paired_diff_mean_sd(df, "brier_score", "mnp", "smnp", data)

            ls_s_m, _, _ = _mean_sd(df, _test_col("log_score", "smnp", data))
            ls_m_m, _, _ = _mean_sd(df, _test_col("log_score", "mnp", data))
            ls_d_m, ls_d_sd, ls_p, _ = _paired_diff_mean_sd(df, "log_score", "mnp", "smnp", data)

            beta_rmse_m, beta_rmse_sd, _ = _mean_sd(df, _recov_experiment_col("beta_rmse", "smnp", data))
            delta_rmse_m, delta_rmse_sd, _ = _mean_sd(df, _recov_experiment_col("delta_rmse", "smnp", data))
            mnp_beta_rmse_m, _, _ = _mean_sd(df, _recov_experiment_col("beta_rmse", "mnp", data))

            lam_m, lam_sd, _ = _mean_sd(df, _derived_recov_col("lambda_star_mean", "smnp", data))
            lam_lo_m, _, _ = _mean_sd(df, _derived_recov_col("lambda_star_sq_q025", "smnp", data))
            lam_hi_m, _, _ = _mean_sd(df, _derived_recov_col("lambda_star_sq_q975", "smnp", data))

            rows.append({
                "sample_size": N,
                "data": data,
                "n_replicates": n_acc,
                "smnp_accuracy": acc_s_m, "smnp_accuracy_sd": acc_s_sd,
                "mnp_accuracy": acc_m_m, "mnp_accuracy_sd": acc_m_sd,
                "accuracy_improvement": acc_d_m, "accuracy_improvement_sd": acc_d_sd,
                "accuracy_improvement_P_gt0": acc_p,
                "smnp_f1": f1_s_m, "mnp_f1": f1_m_m,
                "f1_improvement": f1_d_m, "f1_improvement_sd": f1_d_sd,
                "f1_improvement_P_gt0": f1_p,
                "smnp_brier": br_s_m, "mnp_brier": br_m_m,
                "brier_improvement": br_d_m, "brier_improvement_sd": br_d_sd,
                "brier_improvement_P_gt0": br_p,
                "smnp_log_score": ls_s_m, "mnp_log_score": ls_m_m,
                "log_score_improvement": ls_d_m, "log_score_improvement_sd": ls_d_sd,
                "log_score_improvement_P_gt0": ls_p,
                "smnp_beta_rmse": beta_rmse_m, "smnp_beta_rmse_sd": beta_rmse_sd,
                "mnp_beta_rmse": mnp_beta_rmse_m,
                "smnp_delta_rmse": delta_rmse_m, "smnp_delta_rmse_sd": delta_rmse_sd,
                "smnp_lambda_star": lam_m, "smnp_lambda_star_sd": lam_sd,
                "smnp_lambda_star_sq_q025": lam_lo_m, "smnp_lambda_star_sq_q975": lam_hi_m
            })

    summary_df = pd.DataFrame(rows).sort_values(["data", "sample_size"]).reset_index(drop=True)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "sample_size_summary.csv")
    summary_df.to_csv(out, index=False)
    print(f"[trend] sample-size summary saved to: {out}")
    return summary_df


def create_per_class_recall_summary(all_sample_results, J, save_dir="results_aggregate"):
    rows = []
    for _, cell in all_sample_results.items():
        N = cell["sample_size"]
        rep_csv = os.path.join(cell["output_dir"], "aggregate", "all_replicates.csv")
        if not os.path.exists(rep_csv):
            continue
        df = pd.read_csv(rep_csv)
        for data in ("smnp", "mnp"):
            for c in range(1, J + 1):
                metric = f"recall_class_{c}"
                m, sd, p_gt0, n = _paired_diff_mean_sd(df, metric, "smnp", "mnp", data)
                if n == 0:
                    continue
                rows.append({"sample_size": N, "data": data, "class": c,"recall_diff_mean": m, "recall_diff_sd": sd,
                             "recall_diff_P_gt0": p_gt0, "n_replicates": n})
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["data", "class", "sample_size"]).reset_index(drop=True)
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "per_class_recall_summary.csv")
        out_df.to_csv(out, index=False)
        print(f"per-class recall summary saved to: {out}")
    else:
        print("no per-class recall columns found.")
    return out_df


def print_trend_analysis(summary_df, primary_regime="smnp"):
    print("\nSUMMARY ACROSS SAMPLE SIZES (mean over replicates):")
    print("-" * 64)
    for data_label in summary_df["data"].unique():
        sub = summary_df[summary_df["data"] == data_label]
        print(f"\n  Data regime: {data_label}")
        cols = ["sample_size", "n_replicates", "accuracy_improvement", "accuracy_improvement_sd",
                "accuracy_improvement_P_gt0", "brier_improvement", "smnp_lambda_star", "smnp_lambda_star_sd",
                "smnp_delta_rmse"]
        cols = [c for c in cols if c in sub.columns]
        print(sub[cols].round(4).to_string(index=False))

    prim = summary_df[summary_df["data"] == primary_regime].sort_values("sample_size")
    if len(prim) < 2:
        print("\n[trend] not enough sample sizes for a slope estimate yet.")
        return

    print("\nTREND (primary regime = skewed data):")
    print("-" * 40)
    acc_slope = np.polyfit(prim["sample_size"], prim["accuracy_improvement"], 1)[0]
    print(f"Accuracy improvement slope: {acc_slope:+.6e} per observation "
          f"({'increases' if acc_slope > 0 else 'decreases'} with N)")

    if prim["smnp_lambda_star"].notna().any():
        ls = prim.dropna(subset=["smnp_lambda_star"])
        if len(ls) >= 2:
            ls_slope = np.polyfit(ls["sample_size"], ls["smnp_lambda_star"], 1)[0]
            print(f"lambda_* slope: {ls_slope:+.6e} per observation "
                  f"({'tightens' if abs(ls_slope) < 1e-6 else 'shifts'} with N.")
    if prim["smnp_delta_rmse"].notna().any():
        print("(smnp_delta_rmse is also in sample_size_summary.csv.")

    imp = prim["accuracy_improvement"]
    if (imp > 0).all():
        print("SMNP >= MNP (mean) at every sample size on skewed data.")
    elif (imp < 0).all():
        print("MNP >= SMNP (mean) at every sample size on skewed data.")
    else:
        print("Mixed: SMNP advantage varies with N.")


def _regime_frame(summary_df, data_regime):
    sub = summary_df[summary_df["data"] == data_regime].sort_values("sample_size").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows for data_regime={data_regime!r} in summary_df.")
    return sub


def create_main_trends_plot(summary_df, save_dir="results_aggregate", data_regime="smnp"):
    sub = _regime_frame(summary_df, data_regime)
    N = sub["sample_size"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"SMNP vs MNP across sample sizes (data regime: {data_regime}; "
                 f"points = replicate means, bands = +-1 replicate SD)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(N, sub["smnp_accuracy"], "o-", color="darkblue", label="SMNP")
    ax.fill_between(N, sub["smnp_accuracy"] - sub["smnp_accuracy_sd"],
                    sub["smnp_accuracy"] + sub["smnp_accuracy_sd"], color="darkblue", alpha=0.15)
    ax.plot(N, sub["mnp_accuracy"], "s-", color="darkorange", label="MNP")
    ax.fill_between(N, sub["mnp_accuracy"] - sub["mnp_accuracy_sd"],
                    sub["mnp_accuracy"] + sub["mnp_accuracy_sd"], color="darkorange", alpha=0.15)
    ax.set_xlabel("Sample size (N)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs N"); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.axhline(0, color="grey", lw=1)
    ax.plot(N, sub["accuracy_improvement"], "o-", color="green")
    ax.fill_between(N, sub["accuracy_improvement"] - sub["accuracy_improvement_sd"],
                    sub["accuracy_improvement"] + sub["accuracy_improvement_sd"], color="green", alpha=0.2)
    ax.set_xlabel("Sample size (N)"); ax.set_ylabel("Accuracy improvement (SMNP - MNP)")
    ax.set_title("Paired accuracy improvement vs N"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if sub["smnp_lambda_star"].notna().any():
        lo = np.sqrt(sub["smnp_lambda_star_sq_q025"].clip(lower=0))
        hi = np.sqrt(sub["smnp_lambda_star_sq_q975"].clip(lower=0))
        ax.plot(N, sub["smnp_lambda_star"], "o-", color="purple")
        ax.fill_between(N, lo, hi, color="purple", alpha=0.2)
        ax.set_ylabel("lambda_* (posterior mean, shaded = 95% CI)")
        ax.set_title("Skewness recovery vs N (lambda_*, identified)")
    else:
        ax.text(0.5, 0.5, "no lambda_star\n(regime has delta=0, or _recov columns missing)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Skewness recovery vs N")
    ax.set_xlabel("Sample size (N)"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f"sample_size_trends_{data_regime}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def create_per_class_recall_plot(per_class_df, save_dir="results_aggregate", data_regime="smnp", J=4):
    if per_class_df is None or per_class_df.empty:
        return None
    sub = per_class_df[per_class_df["data"] == data_regime]
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, J, figsize=(4.2 * J, 4.5), sharey=True)
    if J == 1:
        axes = [axes]
    fig.suptitle(f"Per-class recall difference (SMNP - MNP), data regime: {data_regime}", fontsize=14, fontweight="bold")
    for k, c in enumerate(range(1, J + 1)):
        ax = axes[k]
        cd = sub[sub["class"] == c].sort_values("sample_size")
        if cd.empty:
            ax.set_visible(False); continue
        N = cd["sample_size"].to_numpy()
        ax.axhline(0, color="grey", lw=1)
        ax.plot(N, cd["recall_diff_mean"], "o-", color="teal")
        ax.fill_between(N, cd["recall_diff_mean"] - cd["recall_diff_sd"],
                        cd["recall_diff_mean"] + cd["recall_diff_sd"], color="teal", alpha=0.2)
        ax.set_title(f"Alternative {c}")
        ax.set_xlabel("Sample size (N)")
        if k == 0:
            ax.set_ylabel("recall(SMNP) - recall(MNP)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f"per_class_recall_trends_{data_regime}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(all_sample_results, save_dir="results_aggregate", J=4):
    print(f"\n{'=' * 60}\nGENERATING REPLICATION TREND ANALYSIS\n{'=' * 60}")

    summary_df = create_aggregate_summary(all_sample_results, save_dir=save_dir)
    if summary_df.empty:
        print("[trend] no cells with replicate data found; nothing to plot.")
        return summary_df

    print_trend_analysis(summary_df, primary_regime="smnp")

    per_class_df = create_per_class_recall_summary(all_sample_results, J=J, save_dir=save_dir)

    regimes = list(summary_df["data"].unique())
    for regime in regimes:
        try:
            create_main_trends_plot(summary_df, save_dir, data_regime=regime)
        except ValueError as e:
            print(f"[trend] skip main plot for {regime}: {e}")
        create_per_class_recall_plot(per_class_df, save_dir, data_regime=regime, J=J)

    print(f"\n[trend] outputs written under: {save_dir}/")
    return summary_df


def analyze_sample_size_trends(all_sample_results, save_dir="results_aggregate", J=4):
    return generate_all_plots(all_sample_results, save_dir, J=J)


# Misspecification study

def _load_replicates(cell_dir):
    """Load all_replicates.csv for one cell; None if the cell doesn't exist yet."""
    path = os.path.join(cell_dir, "aggregate", "all_replicates.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def analyze_misspecification(scratch_root, run_name="smnp_multinomial", arm="unconstrained",
                             samples_sizes=(300, 500, 1000, 1500), save_dir="results_misspec", misspec_root=None):
    misspec_root = misspec_root or os.path.join(scratch_root, "smnp_misspecification")
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for true_g_dist in ["exponential", "halfnormal"]:
        wrong_g_dist = "halfnormal" if true_g_dist == "exponential" else "exponential"
        for N in samples_sizes:
            matched_dir = os.path.join(scratch_root, run_name, true_g_dist, f"result_N_{N}_{arm}")
            mismatched_dir = os.path.join(misspec_root, true_g_dist, f"result_N_{N}_{arm}_fitas_{wrong_g_dist}")
            m_df = _load_replicates(matched_dir)
            x_df = _load_replicates(mismatched_dir)
            if m_df is None or x_df is None:
                print(f"missing cell (skipped): true={true_g_dist} N={N}  "
                      f"matched={'ok' if m_df is not None else 'MISSING'}  "
                      f"mismatched(fit as {wrong_g_dist})={'ok' if x_df is not None else 'MISSING'}")
                continue

            merged = pd.merge(m_df, x_df, on="replicate", suffixes=("_matched", "_mismatched"))
            if len(merged) == 0:
                print(f"no overlapping replicates for true={true_g_dist} N={N}")
                continue

            row = {"true_g_dist": true_g_dist, "wrong_g_dist": wrong_g_dist,
                   "sample_size": N, "n_replicates": len(merged)}

            metric_specs = [("accuracy", "accuracy", True), ("f1", "f1_weighted", True), ("brier", "brier_score", False), ("log_score", "log_score", False)]
            for label, metric_name, higher_better in metric_specs:
                col = _test_col(metric_name, "smnp", "smnp")
                ca, cb = f"{col}_matched", f"{col}_mismatched"
                if ca not in merged.columns or cb not in merged.columns:
                    continue
                a = pd.to_numeric(merged[ca], errors="coerce")
                b = pd.to_numeric(merged[cb], errors="coerce")
                diff = (a - b) if higher_better else (b - a)
                diff = diff.dropna()
                row[f"{label}_matched_mean"] = a.mean()
                row[f"{label}_mismatched_mean"] = b.mean()
                row[f"{label}_cost_of_misspec_mean"] = diff.mean()
                row[f"{label}_cost_of_misspec_sd"] = diff.std(ddof=1) if len(diff) > 1 else np.nan
                row[f"{label}_P_matched_better"] = float((diff > 0).mean()) if len(diff) else np.nan

            lam_col = _derived_recov_col("lambda_star_mean", "smnp", "smnp")
            la, lb = f"{lam_col}_matched", f"{lam_col}_mismatched"
            if la in merged.columns and lb in merged.columns:
                row["lambda_star_matched_mean"] = pd.to_numeric(merged[la], errors="coerce").mean()
                row["lambda_star_mismatched_mean"] = pd.to_numeric(merged[lb], errors="coerce").mean()

            rows.append(row)

    if not rows:
        print("no cells found -- nothing to summarize")
        return None

    out_df = pd.DataFrame(rows)
    out = os.path.join(save_dir, "misspecification_summary.csv")
    out_df.to_csv(out, index=False)
    print(f"[misspec] wrote {out}  ({len(out_df)} cells)")

    _plot_misspecification(out_df, save_dir)
    return out_df


def _plot_misspecification(df, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    metrics = [("accuracy", "Accuracy"), ("f1", "F1 (weighted)"), ("brier", "Brier score"), ("log_score", "Log-score")]
    colors = {"exponential": "tab:blue", "halfnormal": "tab:red"}
    for ax, (metric, label) in zip(axes, metrics):
        mcol = f"{metric}_cost_of_misspec_mean"
        scol = f"{metric}_cost_of_misspec_sd"
        if mcol not in df.columns:
            ax.set_visible(False)
            continue
        for true_g_dist, sub in df.groupby("true_g_dist"):
            sub = sub.dropna(subset=[mcol]).sort_values("sample_size")
            if sub.empty:
                continue
            ax.plot(sub["sample_size"], sub[mcol], "o-",
                    label=f"truth={true_g_dist}", color=colors.get(true_g_dist))
            if scol in sub.columns:
                ax.fill_between(sub["sample_size"], sub[mcol] - sub[scol], sub[mcol] + sub[scol],
                                alpha=0.15, color=colors.get(true_g_dist))
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(label)
        ax.set_xlabel("Sample size")
        ax.legend()
        ax.grid(False)
    fig.supylabel("Correct - Misspecified Fit")
    fig.suptitle("Effect of misspecifying the mixing distribution", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(save_dir, "misspecification_cost.pdf")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[misspec] wrote {out}")


if __name__ == "__main__":
    print("Replication trend module.")
