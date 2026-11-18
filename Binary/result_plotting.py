"""
This script reads the predictive_ALL_test_summary csv file and plot Figure 1.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# The summary_csv here is obtained by running:
# python aggregate_predictive.py "{args.output_root}/predictive/*/*/*/test_raw.tsv" predictive_ALL_test_summary.csv

summary_csv = "predictive_ALL_test_summary.csv"  # CSV manifest is obtained by running aggregate
output_dir = "binary_predictive_figures"
link_type = "probit"

data_types = ("cloglog", "probit")
model_types = ("skewed", "non_skewed")

metrics = (("accuracy_mean", "Accuracy", "accuracy"), ("AUC_mean", "AUC", "auc"),
           ("f1_score_mean", "F1 score", "f1_score"), ("Log_loss_mean", "Log loss", "log_loss"),
           ("Brier_score_mean", "Brier score", "Brier_score"))
model_labels = {"skewed": "skewed", "non_skewed": "non-skewed"}
model_styles = {"skewed": dict(marker="o", linestyle="-", linewidth=2.2, markersize=6),
                "non_skewed": dict(marker="s", linestyle="--", linewidth=2.2, markersize=6)}

label_fontsize = 25
title_fontsize = 17
tick_fontsize = 17
legend_fontsize = 13

def load_predictive_summary(csv_path=summary_csv, link_type=link_type):
    df = pd.read_csv(csv_path)
    required = {"num_samples", "data_type", "model_type", "link_type"}
    required.update(metric for metric, _, _ in metrics)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("The summary file is missing required columns: " + ", ".join(missing))
    df = df[df["link_type"] == link_type].copy()
    if df.empty:
        raise ValueError(f"No rows found for link_type={link_type!r} in {csv_path}")
    return df


def _plot_models(ax, df, data_type, metric):
    """ Draw skewed and non-skewed model lines for one data type/metric. """
    sample_sizes = sorted(df["num_samples"].unique())
    x_positions = list(range(len(sample_sizes)))
    x_lookup = {n: i for i, n in enumerate(sample_sizes)}
    for model_type in model_types:
        subset = df[(df["data_type"] == data_type) & (df["model_type"] == model_type)].sort_values("num_samples")
        if subset.empty:
            continue
        x = [x_lookup[n] for n in subset["num_samples"]]
        ax.plot(x, subset[metric], label=model_labels[model_type], **model_styles[model_type])

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(False)

    return x_positions, sample_sizes


def plot_combined(df, output_dir=output_dir, filename="binary_predictive_combined"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 8), sharex="col", sharey=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.91, wspace=0.26, hspace=0.08)

    for col_idx, (metric, metric_label, _) in enumerate(metrics):
        ax = axes[0, col_idx]
        x_positions, sample_sizes = _plot_models(ax=ax, df=df, data_type="cloglog", metric=metric)
        ax.set_title(metric_label, fontsize=title_fontsize, pad=10)
        if col_idx == 0:
            ax.set_ylabel("Cloglog Data", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize, frameon=False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.set_xlabel("")

    for col_idx, (metric, _, _) in enumerate(metrics):
        ax = axes[1, col_idx]
        x_positions, sample_sizes = _plot_models(ax=ax, df=df, data_type="probit", metric=metric)
        ax.set_title("")
        if col_idx == 0:
            ax.set_ylabel("Probit Data", fontsize=label_fontsize)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=tick_fontsize)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    base = os.path.join(output_dir, filename)
    saved = []

    path = base + ".pdf"
    fig.supxlabel("Sample Sizes", fontsize=label_fontsize)
    fig.savefig(path, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    return saved


if __name__ == "__main__":
    df = load_predictive_summary()
    combined_files = plot_combined(df)
    for path in combined_files:
        print(f"{path}")
