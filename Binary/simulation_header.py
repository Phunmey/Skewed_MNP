"""
This file creates the headers for the results.
"""

import os
import pandas as pd


def initialize_files(train_f, test_f):
    file1 = open(train_f, "w")
    header = (
        'num_samples\treplicate\tmodel_type\tlink_type\taccuracy\tAUC\tLog_loss\tBrier_score\tAUPRC\t'
        'precision\trecall\tf1_score\telpd_loo\telpd_loo_se\tloo_value\tloo_se\t'
        'confusion_matrix\ttime_taken\n')
    file1.write(header)

    file2 = open(test_f, "w")
    file2.write(header)

    return file1, file2


def append_files(train_f, test_f):
    file1 = open(train_f, "a")
    file2 = open(test_f, "a")
    return file1, file2


def close_files(file1, file2):
    file1.close()
    file2.close()


def load_completed_predictive_replicates(tsv_path):
    if not os.path.exists(tsv_path):
        return set()
    existing = pd.read_csv(tsv_path, sep='\t')
    if existing.empty:
        return set()
    return set(zip(existing['num_samples'], existing['replicate']))


def open_or_resume_predictive_files(train_f, test_f):
    if os.path.exists(train_f) and os.path.exists(test_f):
        return append_files(train_f, test_f)
    return initialize_files(train_f, test_f)


def load_completed_recovery_replicates(out_csv):
    if not os.path.exists(out_csv):
        return set()
    existing = pd.read_csv(out_csv)
    return set(zip(existing['num_samples'], existing['data_type'],
                    existing['model_type'], existing['replicate']))


def append_recovery_row(row: dict, out_csv):
    write_header = not os.path.exists(out_csv)
    pd.DataFrame([row]).to_csv(out_csv, mode='a', header=write_header, index=False)
