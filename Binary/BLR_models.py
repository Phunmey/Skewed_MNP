import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from matplotlib import pyplot as plt
from time import time

from bayesian_plotting import nonskewed_trace, nonskewed_ppc, skewed_trace, skewed_ppc
from performance_metrics import training_perf_metrics, testing_perf_metrics

random_seed = 8927

try:
    az.style.use("arviz-white")
except (ValueError, KeyError):
    pass
plt.rcParams["figure.figsize"] = [7, 6]
plt.rcParams["figure.dpi"] = 100

mu_G = np.sqrt(2 / np.pi)


def inv_cloglog(x):
    return 1 - pt.exp(-pt.exp(x))


def _build_pymc_model(use_features, x_data, y_data, model_type='non_skewed', link_type='logistic'):
    """Shared model-construction logic for both recovery and predictive workflows."""
    coords = {"xvars": use_features}
    with pm.Model(coords=coords) as model:
        x = pm.Data("x", np.asarray(x_data, dtype=float))
        y = pm.Data('y', np.asarray(y_data))

        betas = pm.Normal("betas", 0, sigma=2.5, dims=("xvars",))

        if model_type == 'non_skewed':
            linear_pred = pm.math.dot(x, betas)
        else:
            delta = pm.Normal("delta", 0, 1)
            w = pm.HalfNormal('w', sigma=1, shape=x.shape[0])
            linear_pred = pm.math.dot(x, betas) + (delta * (w - mu_G))

        if link_type == 'logistic':
            probs = pm.Deterministic("probs", pm.math.invlogit(linear_pred))
        elif link_type == 'probit':
            probs = pm.Deterministic("probs", pm.math.invprobit(linear_pred))
        else:
            probs = pm.Deterministic("probs", inv_cloglog(linear_pred))

        pm.Bernoulli("yobs", p=probs, observed=y, shape=probs.shape[0])

    return model


def build_model_recovery(use_features, x_full, y_full, sample, replicate, model_type='non_skewed',
                         link_type='logistic', draws=2000, tune=3000, chains=4, cores=None, mp_ctx='fork',
                         target_accept=0.95, seed=random_seed, save_plots=True, plot_dir='.'):
    print(f"[recovery] building model: n={sample} rep={replicate} {model_type}/{link_type}")
    start = time()
    model = _build_pymc_model(use_features, x_full, y_full, model_type, link_type)
    actual_cores = cores if cores is not None else chains
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=actual_cores, mp_ctx=mp_ctx,
                          target_accept=target_accept, idata_kwargs={'log_likelihood': True}, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    fit_time_seconds = time() - start

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        tag = f"n{sample}_rep{replicate}_{link_type}_{model_type}"
        try:
            if model_type == 'non_skewed':
                nonskewed_trace(trace)
                plt.savefig(f"{plot_dir}/trace_{tag}.png")
                plt.clf()

                nonskewed_ppc(trace)
                plt.savefig(f"{plot_dir}/ppc_{tag}.png")
                plt.clf()
            else:
                skewed_trace(trace)
                plt.savefig(f"{plot_dir}/trace_{tag}.png")
                plt.clf()
                skewed_ppc(trace)
                plt.savefig(f"{plot_dir}/ppc_{tag}.png")
                plt.clf()
        except Exception as e:
            print(f"[recovery] WARNING: plotting failed for {tag} ({e}); fit results are unaffected.")
            plt.clf()
    return model, trace, fit_time_seconds


def build_model_predictive(file1, file2, use_features, xtrain, ytrain, xtest, ytest, sample,
                           replicate, model_type='non_skewed', link_type='logistic',
                           draws=2000, tune=3000, chains=4, cores=None, mp_ctx='fork', target_accept=0.95,
                           seed=random_seed, save_plots=False, plot_dir='.'):
    print(f"predictive model: n={sample} rep={replicate} {model_type}/{link_type}")
    start1 = time()
    model = _build_pymc_model(use_features, xtrain, ytrain, model_type, link_type)
    actual_cores = cores if cores is not None else chains
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=actual_cores, mp_ctx=mp_ctx,
                          target_accept=target_accept,
                          idata_kwargs={'log_likelihood': True}, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
        pm.sample_posterior_predictive(trace, random_seed=seed, extend_inferencedata=True)
    print("inference done")

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        tag = f"n{sample}_rep{replicate}_{link_type}_{model_type}"
        try:
            if model_type == 'non_skewed':
                nonskewed_trace(trace)
                plt.savefig(f"{plot_dir}/trace_{tag}.png")
                plt.clf()

                nonskewed_ppc(trace)
                plt.savefig(f"{plot_dir}/ppc_{tag}.png")
                plt.clf()
            else:
                skewed_trace(trace)
                plt.savefig(f"{plot_dir}/trace_{tag}.png")
                plt.clf()
                skewed_ppc(trace)
                plt.savefig(f"{plot_dir}/ppc_{tag}.png")
                plt.clf()
        except Exception as e:
            print(f"plotting failed for {tag} ({e}).")
            plt.clf()

    train_probs = trace.posterior['probs'].mean(dim=['chain', 'draw'])
    ypred_train = (np.asarray(train_probs) > 0.5)

    (train_acc, train_prec, train_rec, train_f1, train_auc, train_log_loss, train_brier, train_auprc, elpd_loo,
     elpd_loo_se, loo_value, loo_se, train_cm) = training_perf_metrics(ytrain, ypred_train, train_probs, trace)

    end1 = time()
    train_time = end1 - start1

    file1.write(
        f"{sample}\t{replicate}\t{model_type}\t{link_type}\t{train_acc}\t{train_auc}\t{train_log_loss}\t"
        f"{train_brier}\t{train_auprc}\t{train_prec}\t{train_rec}\t{train_f1}\t{elpd_loo}\t{elpd_loo_se}\t"
        f"{loo_value}\t{loo_se}\t{train_cm}\t{train_time}\n")
    file1.flush()

    start2 = time()
    print("testing...")
    with model:
        pm.set_data({"x": np.asarray(xtest, dtype=float)})
        posterior_ds = trace.posterior.to_dataset() if hasattr(trace.posterior, 'to_dataset') else trace.posterior
        if model_type == 'skewed' and 'w' in posterior_ds.data_vars:
            posterior_ds = posterior_ds.drop_vars('w')

        post_predictive = pm.sample_posterior_predictive(posterior_ds, var_names=['probs', 'yobs'], predictions=True,
                                                         random_seed=seed, progressbar=False)

    got_rows = post_predictive.predictions['probs'].shape[-1]
    assert got_rows == len(xtest), (f"Posterior predictive returned {got_rows} rows, expected {len(xtest)} (len(xtest)).")

    test_probs = post_predictive.predictions['probs'].mean(dim=['chain', 'draw'])
    ypred_test = (np.asarray(test_probs) > 0.5)

    (test_acc, test_prec, test_rec, test_f1, test_auc, test_log_loss, test_brier, test_auprc,
     elpd_loo, elpd_loo_se, loo_value, loo_se, test_cm) = testing_perf_metrics(
        ytest, ypred_test, test_probs, trace)

    end2 = time()
    test_time = end2 - start2

    file2.write(
        f"{sample}\t{replicate}\t{model_type}\t{link_type}\t{test_acc}\t{test_auc}\t{test_log_loss}\t"
        f"{test_brier}\t{test_auprc}\t{test_prec}\t{test_rec}\t{test_f1}\t{elpd_loo}\t{elpd_loo_se}\t"
        f"{loo_value}\t{loo_se}\t{test_cm}\t{test_time}\n")
    file2.flush()
    print("finished")

    return trace