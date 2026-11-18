"""
This script reads the raw csv files from the result root and plot Figure 2. Here, the root is cloglog and probit folders.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

recovery_root = Path(r".")

output_dir = recovery_root / "inference_summary"
output_dir.mkdir(parents=True, exist_ok=True)

combined_csv = None

label_fontsize = 16
title_fontsize = 15
tick_fontsize = 13
legend_fontsize = 12

def load_all_raw(root):
    """Read all the raw.csv under results folder."""
    root = Path(root)
    files = sorted(root.rglob("raw.csv"))
    if not files:
        raise FileNotFoundError(f"No raw.csv files were found under:\n{root}\n\n"
                                "recovery_root should be the directory containing the 'cloglog' and 'probit' folders.")

    dfs = []
    for f in files:
        rel = f.relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            print(f"Skipping unexpected path: {f}")
            continue

        data_type = parts[-4]
        model_type = parts[-3]
        link_type = parts[-2]

        d = pd.read_csv(f)
        d["data_type"] = data_type
        d["model_type"] = model_type
        d["link_type"] = link_type
        d["source_file"] = str(f)

        dfs.append(d)

    if not dfs:
        raise RuntimeError("raw.csv files were found, but none matched the expected folder structure.")

    df = pd.concat(dfs, ignore_index=True)
    id_cols = ["num_samples", "data_type", "model_type", "link_type", "replicate"]
    id_cols = [c for c in id_cols if c in df.columns]

    n_dup = df.duplicated(subset=id_cols).sum()
    if n_dup:
        print(f"{n_dup} duplicate replicate row(s) found.")
        df = df.drop_duplicates(subset=id_cols, keep="first")

    return df.reset_index(drop=True)


def load_recovery_data():
    """
    Load either the already-combined CSV or the folder tree.
    This guarantees that df_all is defined exactly once and is then used in the script.
    """
    if combined_csv is not None:
        combined_path = Path(combined_csv)
        if not combined_path.exists():
            raise FileNotFoundError(f"combined_csv does not exist:\n{combined_path}")

        df = pd.read_csv(combined_path)
        return df.reset_index(drop=True)

    return load_all_raw(recovery_root)

def sample_axis(df):
    sizes = sorted(df["num_samples"].dropna().unique())
    positions = np.arange(len(sizes))
    lookup = {n: i for i, n in enumerate(sizes)}
    return sizes, positions, lookup

df_all = load_recovery_data()
df_skewed_probit = df_all[(df_all["model_type"] == "skewed") & (df_all["link_type"] == "probit")].copy()
if "delta_postmean" not in df_skewed_probit.columns:
    raise ValueError("delta_postmean is missing from the skewed-model raw files.")

delta_rows = []
for (data_type, n), g in df_skewed_probit.groupby(["data_type", "num_samples"]):
    vals = g["delta_postmean"].dropna()
    if vals.empty:
        continue
    delta_rows.append(
        {"data_type": data_type, "num_samples": n, "n_replicates": len(vals), "delta_mean": vals.mean(),
         "delta_sd_across_reps": vals.std(ddof=1), "delta_q025_across_reps": vals.quantile(0.025),
         "delta_q975_across_reps": vals.quantile(0.975),
         "delta_avg_postsd": g.loc[vals.index, "delta_postsd"].mean()})

delta_summary = (pd.DataFrame(delta_rows).sort_values(["data_type", "num_samples"]).reset_index(drop=True))
delta_summary.to_csv(output_dir / "delta_summary.csv", index=False)

print("\nDelta Summary")
print(delta_summary.round(4).to_string(index=False))

# Delta figure: across-replicate 2.5%--97.5% interval
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sizes, positions, x_lookup = sample_axis(delta_summary)

for data_type in ["cloglog", "probit"]:
    s = delta_summary[delta_summary["data_type"] == data_type].sort_values("num_samples")

    if s.empty:
        continue
    x = np.asarray([x_lookup[n] for n in s["num_samples"]])
    lower = (s["delta_mean"].to_numpy() - s["delta_q025_across_reps"].to_numpy())
    upper = (s["delta_q975_across_reps"].to_numpy() - s["delta_mean"].to_numpy())

    ax.errorbar(x, s["delta_mean"], yerr=np.vstack([lower, upper]), marker="o", linewidth=2, markersize=6,
                capsize=4,
                label=f"{data_type} data")

ax.axhline(0, linewidth=1, linestyle=":")
ax.set_xticks(positions)
ax.set_xticklabels([str(n) for n in sizes], fontsize=tick_fontsize)
ax.set_xlabel("Sample size", fontsize=label_fontsize)
ax.set_ylabel(r"Posterior mean of $\alpha$", fontsize=label_fontsize)
ax.tick_params(axis="y", labelsize=tick_fontsize)
ax.legend(fontsize=legend_fontsize, frameon=False)
ax.grid(False)

fig.tight_layout()
fig.savefig(output_dir / "delta_by_data_type.pdf")#, bbbox_inches="tight")
plt.close(fig)
