"""
This file averages the predictive metrics across replicates.
"""
import os
import glob
import pandas as pd

METRIC_COLS = ['accuracy', 'AUC', 'Log_loss', 'Brier_score', 'AUPRC', 'precision',
               'recall', 'f1_score', 'elpd_loo', 'loo_value']


def _infer_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 4:
        return None, None, None
    link_part, model_type, data_type = parts[-2], parts[-3], parts[-4]
    link_type = link_part[len('link-'):] if link_part.startswith('link-') else link_part
    return data_type, model_type, link_type


def _read_raw(raw_tsv_or_pattern, sep='\t'):
    if isinstance(raw_tsv_or_pattern, (list, tuple)):
        paths = []
        for p in raw_tsv_or_pattern:
            paths.extend(sorted(glob.glob(p)) if glob.has_magic(p) else [p])
    elif glob.has_magic(raw_tsv_or_pattern):
        paths = sorted(glob.glob(raw_tsv_or_pattern))
    else:
        paths = [raw_tsv_or_pattern]

    if not paths:
        raise FileNotFoundError(f"No files matched: {raw_tsv_or_pattern}")

    dfs = []
    for p in paths:
        d = pd.read_csv(p, sep=sep)
        data_type, model_type, link_type = _infer_from_path(p)
        if data_type is not None:
            d['data_type'] = data_type
            d['model_type'] = model_type
            d['link_type'] = link_type
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)


def aggregate_predictive(raw_tsv_or_pattern, out_csv=None, sep='\t'):
    df = _read_raw(raw_tsv_or_pattern, sep=sep)

    id_cols = [c for c in ['num_samples', 'data_type', 'replicate', 'model_type', 'link_type']
               if c in df.columns]
    n_dupes = df.duplicated(subset=id_cols).sum()
    if n_dupes:
        print(f"{n_dupes} duplicate replicate row(s) found across the matched files.")
        df = df.drop_duplicates(subset=id_cols, keep='first')

    group_cols = [c for c in ['num_samples', 'data_type', 'model_type', 'link_type'] if c in df.columns]
    metric_cols = [c for c in METRIC_COLS if c in df.columns]

    agg = df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
    agg.columns = [f'{col}_{stat}' for col, stat in agg.columns]
    agg = agg.reset_index()
    agg['n_replicates'] = df.groupby(group_cols).size().values

    if out_csv:
        agg.to_csv(out_csv, index=False)
    return agg


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "cloglog_test_result.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "predictive_results_summary.csv"
    summary = aggregate_predictive(raw, out)
    print(summary)