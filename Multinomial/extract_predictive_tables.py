import argparse
import os
import numpy as np
import pandas as pd

run_name = "smnp_multinomial"
arm = "unconstrained"
sample_sizes = [300, 500, 1000, 1500]
metrics = [("accuracy", "Acc.", True), ("f1_macro", "F1 (Macro)", True),
           ("brier_score", "Brier-S", False), ("log_score", "Log-S", False)]


def cell_dir(scratch_root, g_dist, N):
    return os.path.join(scratch_root, run_name, g_dist, f"result_N_{N}_{arm}")


def test_col(metric, model, data):
    return f"test_metrics__{metric}__{model}_on_{data}"


def extract_row(df, N, data_regime):
    row = {"sample_size": N, "data": data_regime, "n_replicates": len(df)}
    for metric, label, higher_better in metrics:
        smnp_col = test_col(metric, "smnp", data_regime)
        mnp_col = test_col(metric, "mnp", data_regime)
        if smnp_col not in df.columns or mnp_col not in df.columns:
            print(f"missing columns for {metric} at N={N}, data={data_regime}: looked for '{smnp_col}' / '{mnp_col}'")
            continue
        smnp_vals = pd.to_numeric(df[smnp_col], errors="coerce")
        mnp_vals = pd.to_numeric(df[mnp_col], errors="coerce")

        row[f"smnp_{metric}_mean"] = smnp_vals.mean()
        row[f"smnp_{metric}_sd"] = smnp_vals.std(ddof=1)
        row[f"mnp_{metric}_mean"] = mnp_vals.mean()
        row[f"mnp_{metric}_sd"] = mnp_vals.std(ddof=1)

        diff = (smnp_vals - mnp_vals) if higher_better else (mnp_vals - smnp_vals)
        diff = diff.dropna()
        row[f"{metric}_p_win"] = float((diff > 0).mean()) if len(diff) else np.nan
        row[f"{metric}_n_paired"] = int(len(diff))
    return row


def build_table(scratch_root, g_dist):
    rows = []
    for N in sample_sizes:
        cdir = cell_dir(scratch_root, g_dist, N)
        rep_csv = os.path.join(cdir, "aggregate", "all_replicates.csv")
        if not os.path.exists(rep_csv):
            print(f"missing cell (skipped): {rep_csv}")
            continue
        df = pd.read_csv(rep_csv)
        print(f"N={N}: {len(df)} replicates loaded from {rep_csv}")
        for data_regime in ["smnp", "mnp"]:
            rows.append(extract_row(df, N, data_regime))
    return pd.DataFrame(rows)


def print_summary(table_df, g_dist):
    print(f"\n=== {g_dist}: mean (SD), both models, both regimes ===\n")
    for N in sample_sizes:
        for data_regime, data_label in [("smnp", "S"), ("mnp", "NS")]:
            sub = table_df[(table_df["sample_size"] == N) & (table_df["data"] == data_regime)]
            if sub.empty:
                continue
            r = sub.iloc[0]

            def fmt(model, metric, ndp=4):
                m, s = r.get(f"{model}_{metric}_mean"), r.get(f"{model}_{metric}_sd")
                return "--" if pd.isna(m) else f"{m:.{ndp}f} ({s:.{ndp}f})"

            print(f"N={N:<5} {data_label:<2} SMNP  Acc={fmt('smnp','accuracy')}  "
                  f"F1={fmt('smnp','f1_macro')}  Brier={fmt('smnp','brier_score')}  "
                  f"LogS={fmt('smnp','log_score')}")
            print(f"N={N:<5} {data_label:<2} MNP   Acc={fmt('mnp','accuracy')}  "
                  f"F1={fmt('mnp','f1_macro')}  Brier={fmt('mnp','brier_score')}  "
                  f"LogS={fmt('mnp','log_score')}")
            pw = lambda m: f"{r.get(f'{m}_p_win'):.2f}" if pd.notna(r.get(f"{m}_p_win")) else "--"
            print(f"N={N:<5} {data_label:<2} p_win Acc={pw('accuracy')}  F1={pw('f1_macro')}  "
                  f"Brier={pw('brier_score')}  LogS={pw('log_score')}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-root", type=str, default=os.environ.get("SCRATCH", "./results_scratch_v2"))
    ap.add_argument("--g-dist", choices=["halfnormal", "exponential"], required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    table_df = build_table(args.scratch_root, args.g_dist)
    out_csv = args.out_csv or f"predictive_table_{args.g_dist}.csv"
    table_df.to_csv(out_csv, index=False)
    print(f"\n Wrote {out_csv}  ({len(table_df)} rows)")

    print_summary(table_df, args.g_dist)


if __name__ == "__main__":
    main()