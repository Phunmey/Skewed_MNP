import matplotlib.pyplot as plt

from aggregate_predictive import aggregate_predictive

results_root = "./results"
link_type = "probit"
data_types = ("cloglog", "probit")
model_types = ("skewed", "non_skewed")
metrics = ("accuracy_mean", "AUC_mean", "f1_score_mean", "Log_loss_mean", "elpd_loo_mean", "elpd_loo_std")
metric_labels = ("Accuracy", "AUC", "F1 score", "Log loss", "ELPD-LOO", "ELPD-LOO SE")


def load_summary(results_root=results_root, link_type=link_type):
    pattern = f"{results_root}/predictive/*/*/{link_type}/test_raw.tsv"
    return aggregate_predictive(pattern)


def plot_figure1(df, data_types=data_types, model_types=model_types,
                 metrics=metrics, metric_labels=metric_labels,
                 link_type=link_type, savepath=None):
    fig, axes = plt.subplots(nrows=len(data_types), ncols=len(metrics), figsize=(4 * len(metrics), 4 * len(data_types)), sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    for row_idx, data_type in enumerate(data_types):
        for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]
            for model in model_types:
                subset = df[(df["data_type"] == data_type) & (df["model_type"] == model)]
                subset = subset.sort_values("num_samples")
                if metric not in subset.columns:
                    continue
                ax.plot(subset["num_samples"], subset[metric], marker='o', label=model)

            if row_idx == 0:
                ax.set_title(label, fontsize=13)
            if col_idx == 0:
                ax.set_ylabel(f"{data_type.capitalize()}-data", fontsize=13)
            if row_idx == len(data_types) - 1:
                ax.set_xlabel("Number of samples", fontsize=11)
            if row_idx == 0 and col_idx == len(metrics) - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=11)

    fig.suptitle(f"Skewed vs. non-skewed binary models, fitted with a {link_type} link ", fontsize=14, y=1.02)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
        print(f"Saved to {savepath}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    summary = load_summary()
    plot_figure1(summary, savepath="figure1.png")