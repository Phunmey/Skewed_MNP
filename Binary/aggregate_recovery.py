import os
import glob
import pandas as pd


def _infer_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 4:
        return None, None, None
    link_part, model_type, data_type = parts[-2], parts[-3], parts[-4]
    link_type = link_part[len('link-'):] if link_part.startswith('link-') else link_part
    return data_type, model_type, link_type


def _read_raw(raw_csv_or_pattern):
    if isinstance(raw_csv_or_pattern, (list, tuple)):
        paths = []
        for p in raw_csv_or_pattern:
            paths.extend(sorted(glob.glob(p)) if glob.has_magic(p) else [p])
    elif glob.has_magic(raw_csv_or_pattern):
        paths = sorted(glob.glob(raw_csv_or_pattern))
    else:
        paths = [raw_csv_or_pattern]

    if not paths:
        raise FileNotFoundError(f"No files matched: {raw_csv_or_pattern}")

    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        data_type, model_type, link_type = _infer_from_path(p)
        if data_type is not None:
            d['data_type'] = data_type
            d['model_type'] = model_type
            d['link_type'] = link_type
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)


def aggregate_recovery(raw_csv_or_pattern, out_csv=None):
    df = _read_raw(raw_csv_or_pattern)

    id_cols = ['num_samples', 'data_type', 'model_type', 'link_type', 'replicate']
    id_cols = [c for c in id_cols if c in df.columns]
    n_dupes = df.duplicated(subset=id_cols).sum()
    if n_dupes:
        print(f"{n_dupes} duplicate replicate row(s) found across the matched files.")
        df = df.drop_duplicates(subset=id_cols, keep='first')

    beta_cols = sorted({c.rsplit('_', 1)[0] for c in df.columns if c.startswith('beta') and '_true' in c})
    group_cols = ['num_samples', 'data_type', 'model_type', 'link_type']

    rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['n_replicates'] = len(g)
        for b in beta_cols:
            true_col, mean_col, sd_col = f'{b}_true', f'{b}_postmean', f'{b}_postsd'
            row[f'{b}_true'] = g[true_col].iloc[0]
            row[f'{b}_avg_postmean'] = g[mean_col].mean()
            row[f'{b}_sd_across_reps'] = g[mean_col].std()
            row[f'{b}_bias'] = g[mean_col].mean() - g[true_col].iloc[0]
            row[f'{b}_avg_postsd'] = g[sd_col].mean()  # contraction diagnostic
        if 'delta_postmean' in g.columns:
            row['delta_avg_postmean'] = g['delta_postmean'].mean()
            row['delta_sd_across_reps'] = g['delta_postmean'].std()
            row['delta_avg_postsd'] = g['delta_postsd'].mean()
        if 'fit_time_seconds' in g.columns:
            row['fit_time_seconds_mean'] = g['fit_time_seconds'].mean()
            row['fit_time_seconds_total'] = g['fit_time_seconds'].sum()
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "recovery_results_raw.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "recovery_results_summary.csv"
    summary = aggregate_recovery(raw, out)
    cols_to_show = [c for c in summary.columns if 'beta0' in c or c in
                     ('num_samples', 'data_type', 'model_type', 'n_replicates')]
    print(summary[cols_to_show])