import os
import argparse
import warnings

from data_generation import generate_simulated_data, true_beta, feature_names
from data_split import split_data, scale_full_data
from BLR_models import build_model_predictive, build_model_recovery
from performance_metrics import recovery_summary
from simulation_header import (close_files, load_completed_recovery_replicates, append_recovery_row,
                               load_completed_predictive_replicates, open_or_resume_predictive_files)

from aggregate_recovery import aggregate_recovery
from aggregate_predictive import aggregate_predictive

warnings.filterwarnings('ignore')

n_grid = (50, 100, 200, 500, 1000, 3000)
data_types = ('cloglog', 'probit')
model_types = ('skewed', 'non_skewed')
link_types = ('probit', 'cloglog')
base_seed = 8927
n_replicates = 100


def _ensure_dir(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def main_predictive(output_root, n_grid=n_grid, data_types=data_types, model_types=model_types,
                    link_types=link_types, n_replicates=n_replicates, draws=2000, tune=3000, chains=4,
                     cores=None, mp_ctx='fork', base_seed=base_seed, save_plots=False):
    for data_type in data_types:
        for modeltype in model_types:
            for linktype in link_types:
                out_dir = _ensure_dir(output_root, 'predictive', data_type, modeltype, linktype)
                train_f = os.path.join(out_dir, 'train_raw.tsv')
                test_f = os.path.join(out_dir, 'test_raw.tsv')
                done = load_completed_predictive_replicates(test_f)

                file1, file2 = open_or_resume_predictive_files(train_f, test_f)
                try:
                    for sample in n_grid:
                        for replicate in range(n_replicates):
                            if (sample, replicate) in done:
                                print(f"{data_type}/{modeltype}/{linktype} n={sample} rep={replicate} -- already done, skipping")
                                continue

                            seed = base_seed + 1000 * replicate
                            df = generate_simulated_data(sample, model=data_type, seed=seed)
                            df_num = df[feature_names]
                            df_y = df['y']

                            try:
                                xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_num, df_y)
                            except ValueError as e:
                                print(f"Skipping n={sample} {data_type}/{modeltype} rep={replicate}: {e}")
                                continue

                            plot_dir = os.path.join(out_dir, f'n{sample}_rep{replicate}')
                            build_model_predictive(file1, file2, feature_names, xtrain, ytrain, xtest, ytest,
                                                   sample, replicate, model_type=modeltype, link_type=linktype,
                                                   draws=draws, tune=tune, chains=chains, cores=cores, mp_ctx=mp_ctx,
                                                   seed=seed, save_plots=save_plots, plot_dir=plot_dir)

                finally:
                    close_files(file1, file2)

                if os.path.exists(test_f):
                    aggregate_predictive(train_f, out_csv=os.path.join(out_dir, 'train_summary.csv'))
                    aggregate_predictive(test_f, out_csv=os.path.join(out_dir, 'test_summary.csv'))
                    print(f"{data_type}/{modeltype}/{linktype} done -- {out_dir}/")


def main_recovery(output_root, n_grid=n_grid, data_types=data_types, model_types=model_types,
                  link_types=link_types, n_replicates=n_replicates, draws=2000, tune=3000, chains=4,
                  cores=None, mp_ctx='fork', base_seed=base_seed, save_plots=False):
    for data_type in data_types:
        true_beta_ = true_beta[data_type]
        for modeltype in model_types:
            for linktype in link_types:
                out_dir = _ensure_dir(output_root, 'recovery', data_type, modeltype, linktype)
                out_csv = os.path.join(out_dir, 'raw.csv')
                done_already = load_completed_recovery_replicates(out_csv)

                for sample in n_grid:
                    for replicate in range(n_replicates):
                        if (sample, data_type, modeltype, replicate) in done_already:
                            print(f"n={sample} {data_type}/{modeltype} rep={replicate} -- already done, skipping")
                            continue

                        seed = base_seed + 1000 * replicate
                        df = generate_simulated_data(sample, model=data_type, seed=seed)
                        x_full = df[feature_names].to_numpy(dtype=float)
                        y_full = df['y'].to_numpy()
                        #x_full_scaled = scale_full_data(x_full)

                        plot_dir = os.path.join(out_dir, f'n{sample}_rep{replicate}')
                        _, trace, fit_time_seconds = build_model_recovery(feature_names, x_full, y_full, sample,
                                                                          replicate, model_type=modeltype,
                                                                          link_type=linktype, draws=draws, tune=tune,
                                                                          chains=chains, cores=cores, mp_ctx=mp_ctx, seed=seed,
                                                                          save_plots=save_plots, plot_dir=plot_dir)

                        rec = recovery_summary(trace, true_beta_, modeltype)
                        row = {'num_samples': sample, 'data_type': data_type, 'model_type': modeltype,
                               'link_type': linktype, 'replicate': replicate, 'fit_time_seconds': fit_time_seconds}
                        row.update(rec)
                        append_recovery_row(row, out_csv)
                        print(f"{data_type}/{modeltype}/{linktype} n={sample} rep={replicate} "
                              f"beta0_postmean={rec['beta0_postmean']:.3f} beta0_postsd={rec['beta0_postsd']:.3f} "
                              f"fit_time={fit_time_seconds:.1f}s")

                if os.path.exists(out_csv):
                    aggregate_recovery(out_csv, out_csv=os.path.join(out_dir, 'summary.csv'))
                    print(f"{data_type}/{modeltype}/{linktype} done - {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a part of the binary skewed-probit sweep.")
    parser.add_argument('--experiment', choices=['recovery', 'predictive', 'both'], default='both')
    parser.add_argument('--data-types', nargs='+', default=data_types, choices=data_types)
    parser.add_argument('--link-types', nargs='+', default=link_types)
    parser.add_argument('--model-types', nargs='+', default=model_types, choices=model_types)
    parser.add_argument('--n-values', nargs='+', type=int, default=n_grid)
    parser.add_argument('--n-replicates', type=int, default=n_replicates)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--tune', type=int, default=3000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--output-root', default='./results')
    parser.add_argument('--save-plots', action='store_true', help="Save trace/ppc diagnostic figures per run.")
    parser.add_argument('--base-seed', type=int, default=base_seed, help="Base RNG seed")
    parser.add_argument('--cores', type=int, default=None, help="Parallel processes for pm.sample.")
    parser.add_argument('--mp-ctx', default='fork', help="Multiprocessing context.")

    args = parser.parse_args()
    mp_ctx = args.mp_ctx if args.mp_ctx else None

    common = dict(n_grid=args.n_values, data_types=args.data_types, model_types=args.model_types,
                  link_types=args.link_types, n_replicates=args.n_replicates,
                  draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores, mp_ctx=mp_ctx,
                   base_seed=args.base_seed, save_plots=args.save_plots)

    if args.experiment in ('predictive', 'both'):
        main_predictive(args.output_root, **common)

    if args.experiment in ('recovery', 'both'):
        main_recovery(args.output_root, **common)

    print(f"\nDone. Each (data_type, model_type, link_type) combination has its own folder under "
          f"{args.output_root}/{{recovery,predictive}}/, e.g.:")
    print(f"  {args.output_root}/recovery/cloglog/skewed/probit/summary.csv")
    print(f"  {args.output_root}/predictive/cloglog/skewed/probit/test_summary.csv")
    if args.save_plots:
        print(f"  {args.output_root}/recovery/cloglog/skewed/link-probit/n200_rep00/trace_....png")
    print(f"\nTo combine everything across combinations into one master file:")
    print(f'  python aggregate_recovery.py "{args.output_root}/recovery/*/*/*/raw.csv" recovery_ALL_summary.csv')
    print(
        f'  python aggregate_predictive.py "{args.output_root}/predictive/*/*/*/test_raw.tsv" predictive_ALL_test_summary.csv')
