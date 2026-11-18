---
output:
  pdf_document: default
  html_document: default
---

# Binary Analysis: Code and Reproducibility Guide

This folder contains the code used for the binary-response simulation experiments. The binary workflow studies skewed and 
non-skewed binary regression models under probit and cloglog settings. The main driver is `main_binary.py`.

## 1. Main workflow

The main script is ``main_binary.py``. It supports both the recovery and predictive experiments. The recovery experiment
fits the model to the full simulated dataset and evaluates parameter recovery while the predictive experiment performs train/test 
analysis and evaluates predictive performance. Use ``--experiment both`` to run both experiments.

The default simulation grid in `main_binary.py` contains sample sizes ``n = 50, 100, 200, 500, 1000, 3000``. The code considers
``data_types: cloglog, probit, model_types: skewed, non_skewed, link_types: probit, cloglog, and 100 replicates``.

## 2. Software requirements

The needed Python packages are included in the ``requirements.txt`` file.


## 3. Quick smoke test

The following reduced run can be used to confirm that the binary workflow executes successfully.
They are not intended to reproduce the reported simulation results.

```bash
python main_binary.py --experiment both --data-types probit --model-types skewed 
--link-types probit --n-values 50 --n-replicates 1 --draws 500 --tune 500 
--chains 2 --cores 2 --output-root ./example_results
```

## 4. Full binary simulation

The study is organized by combinations of ``data_type x model_type x link_type``.
A full implementation can be run using a command of the form

```bash
python main_binary.py --experiment both --data-types cloglog --model-types skewed 
--link-types probit --n-replicates 100 --draws 5000 --tune 5000 --chains 4 
--cores 4 --output-root ./results
```

The remaining combinations can be run by changing the ``--data-types --model-types --link-types``.


## 5. Random seeds

The default base seed in `main_binary.py` is ``8927``. For replicate `r`, 
the code uses ``seed = base_seed + 1000 * r``. This allows the simulated datasets to be reproduced.

## 6. Parameter Recovery experiment

The recovery workflow is: 
```text
main_binary.py -> generate_simulated_data(...) -> 
build_model_recovery(...) -> recovery_summary(...) -> append results to raw.csv -> 
aggregate_recovery.py -> summary.csv
```

The ``summary.csv`` files contain results for each specific combination of 
``datatype, modeltype and linktype``. 

## 7. Predictive experiment

The predictive workflow is:
```text
main_binary.py -> generate_simulated_data(...) -> train/test split -> 
build_model_predictive(...) -> train and test metrics -> append results to raw.tsv -> 
aggregate_predictive.py -> summary.csv
```

The raw predictive outputs are stored as ``train_raw.tsv, test_raw.tsv`` within the corresponding result folder.
The aggregated outputs are ``train_summary.csv, test_summary.csv``. 

## 8. Main scripts

### `main_binary.py`

This script is the main experiment driver. It runs the parameter-recovery and predictive simulation experiments.

### `data_generation.py`

This script generates the simulated binary datasets and contains the true regression coefficients used for the recovery study.

### `data_split.py`

This script handles train/test splitting and data preparation for the predictive experiment.

### `BLR_models.py`

Defines and fits the Bayesian binary regression models. It contains the skewed and non-skewed model implementations and the model-fitting functions used for recovery and prediction.

### `performance_metrics.py`

Computes parameter-recovery summaries and predictive performance measures.

### `simulation_header.py`

Creates and updates the result files.

### `aggregate_recovery.py`

Reads the raw recovery results and produces summaries across simulation replications.

### `aggregate_predictive.py`

Reads the raw train/test predictive results and produces summaries across simulation replications.

### `bayesian_plotting.py`

Contains plotting functions used for optional posterior diagnostic figures.

### `result_plotting.py`

This script reproduces manuscript's Figure 1.

### `inference_plotting.py`

This script reproduces manuscript's Figure 2.

## 9. Output structure

A typical output structure is:

```text
results/
|-- recovery/
|   |-- cloglog/
|   |   |-- skewed/
|   |   `-- non_skewed/
|   `-- probit/
|       |-- skewed/
|       `-- non_skewed/
|
`-- predictive/
    |-- cloglog/
    |   |-- skewed/
    |   `-- non_skewed/
    `-- probit/
        |-- skewed/
        `-- non_skewed/
```

Each branch is further separated by fitted link type. The raw per-replicate files are the intermediate files needed to reproduce the final aggregated summaries.

## 10. Reproducing reported binary results

To reproduce the binary simulation results:

1. Run all required `(data_type, model_type, link_type)` cells.
2. Use the same sample-size grid and number of replications used for the manuscript analysis.
3. Retain the raw recovery and predictive files.
4. Use the aggregation scripts to produce the final summaries.
5. Use those summaries to construct the reported binary plots.

Note: Our analysis was computed on a computational cluster.
