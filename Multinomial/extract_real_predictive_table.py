import argparse
import os
import numpy as np
import pandas as pd

metrics = [("accuracy", "Acc.", True), ("f1_macro", "F1 (Macro)", True),
           ("brier_score", "Brier-S", False), ("log_score", "Log-S", False)]


def test_col(metric, model):
    return f"test_metrics__{metric}__{model}_on_real"


def extract_row(df):
    row = {"n_repetitions": len(df)}
    for metric, label, higher_better in metrics:
        smnp_col = test_col(metric, "smnp")
        mnp_col = test_col(metric, "mnp")
        if smnp_col not in df.columns or mnp_col not in df.columns:
            print(f"missing columns for {metric}: looked for '{smnp_col}' / '{mnp_col}'")
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


def build_table(out_dir, g_dist):
    rep_csv = os.path.join(out_dir, g_dist, "aggregate", "all_splits.csv")
    if not os.path.exists(rep_csv):
        raise SystemExit(f"Missing {rep_csv} -- run --aggregate-only for this g_dist first")
    df = pd.read_csv(rep_csv)
    print(f"g_dist={g_dist}: {len(df)} repetitions loaded from {rep_csv}")
    return pd.DataFrame([extract_row(df)])


def print_summary(table_df, g_dist):
    print(f"\n=== real data, {g_dist}: mean (SD), SMNP vs MNP, over {table_df.iloc[0]['n_repetitions']:.0f} "
         f"repeated train/test splits ===\n")
    r = table_df.iloc[0]

    def fmt(model, metric, ndp=4):
        m, s = r.get(f"{model}_{metric}_mean"), r.get(f"{model}_{metric}_sd")
        return "--" if pd.isna(m) else f"{m:.{ndp}f} ({s:.{ndp}f})"

    print(f"SMNP  Acc={fmt('smnp', 'accuracy')}  F1={fmt('smnp', 'f1_macro')}  "
         f"Brier={fmt('smnp', 'brier_score')}  LogS={fmt('smnp', 'log_score')}")
    print(f"MNP   Acc={fmt('mnp', 'accuracy')}  F1={fmt('mnp', 'f1_macro')}  "
         f"Brier={fmt('mnp', 'brier_score')}  LogS={fmt('mnp', 'log_score')}")
    pw = lambda m: f"{r.get(f'{m}_p_win'):.2f}" if pd.notna(r.get(f"{m}_p_win")) else "--"
    print(f"p_win Acc={pw('accuracy')}  F1={pw('f1_macro')}  Brier={pw('brier_score')}  LogS={pw('log_score')}")
    print("\n(cross-check: these p_win values should match P(diff>0) in "
         "aggregate/smnp_vs_mnp_paired_test.csv for the same metrics)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=".",
                    help="the exact --out-dir run_real.py used for this study")
    ap.add_argument("--g-dist", choices=["halfnormal", "exponential"], required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    table_df = build_table(args.out_dir, args.g_dist)
    out_csv = args.out_csv or f"predictive_table_real_{args.g_dist}.csv"
    table_df.to_csv(out_csv, index=False)
    print(f"\n[ Wrote {out_csv}")

    print_summary(table_df, args.g_dist)


if __name__ == "__main__":
    main()
