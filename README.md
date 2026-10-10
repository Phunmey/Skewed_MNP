---
output:
  pdf_document: default
  html_document: default
---
# SMNP Code Guide

This repository contains the Python scripts used to implement the Skewed Multinomial Probit (SMNP) model. The files are organized so that the main sampler calls smaller scripts for each Gibbs sampling step.

## Main model files

### `smnp_implementation.py` 
This is the main SMNP sampler class that brings together the Gibbs sampling steps and controls the overall model fitting process. The same class can also be used to fit the standard multinomial probit model by turning off the skewness component.

### `parameter_initialization.py`
This defines the default prior settings and initializes the model parameters before Gibbs sampling begins.

### `w_initialization.py`
This creates initial values for the latent utility matrix `W`.

### `sample_w.py`
This updates the latent utilities `W` during Gibbs sampling.

### `g_sampling.py`
This updates the shared latent mixing variable `g_i` for each observation. The code supports the mixing distribution used in the model, such as half-normal or exponential.

### `beta_sampling.py`
This updates the regression coefficient matrix `beta`.

### `delta.py`
This updates the skewness loading vector `delta`. The parameter controls how strongly the shared latent mixing variable affects the non-reference alternative.

### `psi_construction.py`
This handles the covariance structure of the latent utilities. It updates the covariance-related parameters and reconstructs the full covariance matrix `Psi`.

### `truncated_normal.py`
This contains the helper functions used for drawing from truncated normal distribution when sampling the latent utilities and latent mixing variables.

### `sampling_gibbs.py`
This runs the Gibbs sampler.

## Simulation files

### `data_generation.py`
This generates simulated multinomial data from both the SMNP and standard MNP models.

### `run_sim.py`
This is the main script for running the multinomial simulation study.

### `replication_runner.py`
This script runs repeated simulation experiments and also combines the results across replications.

## Diagnostics and prediction

### `smnp_with_diagnostics.py`
This fits the model and connects the sampler with the diagnostic and prediction procedures. It is used by the replication code to run the model and collect the required output.

### `smnp_diagnostics.py`
This computes posterior and predictive diagnostics.

## Post-processing files

### `postprocess_smnp_recovery.py`
This script reads saved posterior draws after the simulation has finished. It summarizes parameter recovery for quantities such as beta, delta, Psi and the derived skewness measures. It also computes convergence summaries such as R-hat and effective sample size.

### `run_all_postprocess.py`
This runs `postprocess_smnp_recovery.py` across all simulation cells. It then combines the cell-level recovery summaries into consolidated output files.

## Real data analysis

### `run_real.py`
This is the main script for the real-data analysis.It loads the processed real dataset and compares the SMNP and MNP models on the observed data.
The repeated train-test analysis is used for predictive performance, while the full-data analysis is used for inference on the model parameters.

