import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import arviz as az

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score,
                             precision_score, recall_score)
import warnings

warnings.filterwarnings("ignore")


def _as_float_array(a):
    arr = np.asarray(a)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        arr = np.asarray(arr.tolist(), dtype=float)
    return np.ascontiguousarray(arr, dtype=float)


def _safe_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _az_from_posterior(posterior):
    idata = az.from_dict(posterior=posterior)
    names = set(getattr(idata, "posterior", idata).data_vars)
    if names != set(posterior):
        raise RuntimeError(
            f"ArviZ built the wrong variables: got {sorted(names)}, "
            f"expected {sorted(posterior)}. The from_dict calling convention "
            f"does not match this ArviZ version.")
    return idata


def _az_summary(idata, var_names=None, prob=0.95, kind="all", round_to=3):
    out = az.summary(idata, var_names=var_names, hdi_prob=prob, kind=kind, round_to=round_to)

    lo = [c for c in out.columns if c.startswith("hdi_") or c.endswith("_lb") or c.endswith("_lower")]
    hi = [c for c in out.columns if c.startswith("hdi_") or c.endswith("_ub") or c.endswith("_upper")]
    if len(lo) >= 2 and lo == hi:
        lo, hi = [lo[0]], [lo[1]]
    if lo and hi:
        out = out.rename(columns={lo[0]: "ci_lower", hi[0]: "ci_upper"})
    return out


class SMNPDiagnostics:
    def __init__(self, sampler, samples_list, y_true, X_test=None, y_test=None, default_var_names=None):
        self.sampler = sampler
        self.y_true = y_true
        self.X_test = X_test
        self.y_test = y_test
        self.n_chains = len(samples_list)

        if not samples_list:
            raise ValueError("No samples provided.")

        self.n_samples = samples_list[0]['beta'].shape[0]

        self.samples_list = []
        for chain in samples_list:
            flat = chain.copy()
            if flat['beta'].ndim == 3:
                n, j_minus_1, p = flat['beta'].shape
                flat['beta'] = flat['beta'].reshape(n, j_minus_1 * p)
            self.samples_list.append(flat)

        self.combined_samples = self._combine_chains(preserve_chains=False)
        self.default_var_names = default_var_names or ['beta', 'delta', 'gamma']
        self.param_names = self._get_parameter_names()
        self.param_indices = self._get_param_indices()
        self.active_params = self._get_active_params()

        self._idata = self._to_arviz_idata()

    def _param_draws(self, param_name):
        return _as_float_array(self._idata.posterior[param_name].values).ravel()

    def _to_arviz_idata(self):
        posterior = {}

        for k, name in enumerate(self._beta_scalar_names()):
            posterior[name] = np.stack([chain['beta'][:, k] for chain in self.samples_list], axis=0)

        if (self.sampler.include_skewness and 'delta' in self.samples_list[0] and 'delta' in self.default_var_names):
            for j in range(self.sampler.J_minus_1):
                name = f'delta_{j + 2}'
                posterior[name] = np.stack([chain['delta'][:, j] for chain in self.samples_list], axis=0)

        if (self.sampler.J_minus_2 > 0 and 'gamma' in self.samples_list[0] and 'gamma' in self.default_var_names):
            for i in range(self.sampler.J_minus_2):
                name = f'gamma_2_{i + 2}'
                posterior[name] = np.stack([chain['gamma'][:, i] for chain in self.samples_list], axis=0)

        posterior = {k: _as_float_array(v) for k, v in posterior.items()}

        return _az_from_posterior(posterior)

    def _beta_scalar_names(self):
        names = []
        for j in range(2, self.sampler.J + 1):
            for k in range(self.sampler.p):
                names.append(f'beta_{j}_{k}')
        return names

    def _combine_chains(self, preserve_chains=True):
        combined = {}
        for key in self.samples_list[0]:
            values = [s[key] for s in self.samples_list]
            if not hasattr(values[0], 'shape'):
                continue
            if values[0].ndim == 0:
                combined[key] = np.array(values)
            else:
                combined[key] = (np.stack(values, axis=0) if preserve_chains else np.concatenate(values, axis=0))
        return combined

    def _get_parameter_names(self):
        names = []
        for j in range(2, self.sampler.J + 1):
            for k in range(self.sampler.p):
                if k == 0:
                    names.append(f'intercept_{j}')
                else:
                    names.append(f'beta_{j}_{k}')
        if self.sampler.include_skewness:
            for j in range(2, self.sampler.J + 1):
                names.append(f'delta_{j}')
        if self.sampler.J_minus_2 > 0:
            for i in range(2, self.sampler.J):
                names.append(f'gamma_2_{i}')
        return names

    def _get_param_indices(self):
        indices, idx = {}, 0
        indices['beta'] = list(range(idx, idx + self.sampler.p_total))
        idx += self.sampler.p_total
        if getattr(self.sampler, 'include_skewness', False):
            indices['delta'] = list(range(idx, idx + self.sampler.J_minus_1))
            idx += self.sampler.J_minus_1
        else:
            indices['delta'] = []
        indices['gamma'] = (list(range(idx, idx + self.sampler.J_minus_2))
                            if self.sampler.J_minus_2 > 0 else [])
        return indices

    def _get_active_params(self):
        active = {}
        if 'beta' in self.default_var_names:
            active['beta'] = {'indices': self.param_indices['beta'],
                              'names': [self.param_names[i] for i in self.param_indices['beta']], 'samples_key': 'beta'}
        if ('delta' in self.default_var_names and getattr(self.sampler, 'include_skewness', False) and 'delta' in
                self.samples_list[0] and self.param_indices['delta']):
            active['delta'] = {'indices': self.param_indices['delta'],
                               'names': [self.param_names[i] for i in self.param_indices['delta']],
                               'samples_key': 'delta'}
        if ('gamma' in self.default_var_names and self.sampler.J_minus_2 > 0 and 'gamma' in self.samples_list[0] and
                self.param_indices['gamma']):
            active['gamma'] = {'indices': self.param_indices['gamma'],
                               'names': [self.param_names[i] for i in self.param_indices['gamma']],
                               'samples_key': 'gamma'}
        return active

    def convergence_diagnostics(self):
        var_names = list(self._idata.posterior.data_vars)
        az_summary = _az_summary(self._idata, var_names=var_names, prob=0.95, kind='diagnostics', round_to=3)

        rows = []
        for param_name, row in az_summary.iterrows():
            rows.append({
                'parameter': param_name,
                'median': float(np.median(self._param_draws(param_name))),
                'rhat': float(_safe_float(row.get('r_hat', np.nan))),
                'ess': float(_safe_float(row.get('ess_bulk', np.nan))),
                'ess_tail': float(_safe_float(row.get('ess_tail', np.nan))),
                'mcse_mean': float(_safe_float(row.get('mcse_mean', np.nan))),
                'ess_per_sample': float(_safe_float(row.get('ess_bulk', np.nan))) / (self.n_chains * self.n_samples)
            })

        return pd.DataFrame(rows)

    def summarize(self, credible_interval=0.95):
        var_names = list(self._idata.posterior.data_vars)
        az_summary = _az_summary(self._idata, var_names=var_names, prob=credible_interval, kind='all', round_to=3)

        hdi_lower = 'ci_lower' if 'ci_lower' in az_summary.columns else None
        hdi_upper = 'ci_upper' if 'ci_upper' in az_summary.columns else None

        lower_label = f'{(1 - credible_interval) / 2:.1%}'
        upper_label = f'{1 - (1 - credible_interval) / 2:.1%}'

        rows = []
        for param_name, row in az_summary.iterrows():
            r = {
                'parameter': param_name,
                'mean': float(_safe_float(row.get('mean', np.nan))),
                'sd': float(_safe_float(row.get('sd', np.nan))),
                'median': float(np.median(self._param_draws(param_name))),
                'mcse_mean': float(_safe_float(row.get('mcse_mean', np.nan))),
                lower_label: float(row[hdi_lower]) if hdi_lower else np.nan,
                upper_label: float(row[hdi_upper]) if hdi_upper else np.nan,
                'rhat': float(_safe_float(row.get('r_hat', np.nan))),
                'ess': float(_safe_float(row.get('ess_bulk', np.nan))),
                'ess_tail': float(_safe_float(row.get('ess_tail', np.nan)))
            }
            rows.append(r)

        df = pd.DataFrame(rows)

        return df

    def trace_plots(self, n_cols=3, figsize=None, save_path=None):
        posterior = self._idata.posterior

        trace_items = []

        for var_name in posterior.data_vars:
            da = posterior[var_name]
            param_dims = [d for d in da.dims if d not in ["chain", "draw"]]
            if len(param_dims) == 0:
                trace_items.append((var_name, da))
            else:
                shape = [da.sizes[d] for d in param_dims]
                for idx in np.ndindex(*shape):
                    isel_dict = dict(zip(param_dims, idx))
                    da_element = da.isel(isel_dict)
                    suffix = "_".join(str(i) for i in idx)
                    param_name = f"{var_name}_{suffix}"
                    trace_items.append((param_name, da_element))

        n_params = len(trace_items)
        n_rows = int(np.ceil(n_params / n_cols))

        if figsize is None:
            figsize = (18, 2.2 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, (param_name, da) in zip(axes, trace_items):
            for chain in range(da.sizes["chain"]):
                values = da.isel(chain=chain).values
                ax.plot(values, linewidth=0.8, alpha=0.8)

            ax.set_title(param_name, fontsize=9)
            ax.set_xlabel("Draw")
            ax.set_ylabel("Value")
        for ax in axes[n_params:]:
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, axes

    def posterior_plots(self, figsize=(18, 10), true_values=None, save_path=None):
        var_names = list(self._idata.posterior.data_vars)
        ref_vals = None
        if true_values is not None:
            ref_vals = [true_values.get(v, None) for v in var_names]

        axes = az.plot_posterior(self._idata, var_names=var_names, hdi_prob=0.95, ref_val=ref_vals, figsize=figsize)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        return axes

    def autocorrelation_plots(self, max_lag=None, figsize=None, save_path=None, ylim=(-0.1, 1.05), combined=False):
        all_vars = list(self._idata.posterior.data_vars)
        groups = {'beta': [v for v in all_vars if v.startswith('beta')],
                  'delta': [v for v in all_vars if v.startswith('delta')],
                  'gamma': [v for v in all_vars if v.startswith('gamma')]}

        default_lags = {'beta': 200, 'delta': 200, 'gamma': 200}

        def lag_for(group_name):
            if max_lag is None:
                return default_lags[group_name]
            if isinstance(max_lag, dict):
                return max_lag.get(group_name, default_lags[group_name])
            return int(max_lag)

        results = {}
        for group_name, group_vars in groups.items():
            if not group_vars:
                continue

            n_vars = len(group_vars)
            per_row = 2.4 if group_name == 'beta' else 2.0
            auto_height = max(4, n_vars * per_row)
            fs = figsize if figsize is not None else (18, auto_height)

            axes = az.plot_autocorr(self._idata, var_names=group_vars, max_lag=lag_for(group_name), combined=combined,
                                    figsize=fs)

            if ylim is not None:
                axes_flat = np.atleast_1d(axes).ravel()
                for ax in axes_flat:
                    ax.set_ylim(*ylim)

            plt.tight_layout()
            plt.suptitle(
                f'Autocorrelation Functions - {group_name} '
                f'({"combined" if combined else "per chain"}, max_lag={lag_for(group_name)})', y=1.02, fontsize=13)

            if save_path:
                base, ext = os.path.splitext(save_path)
                plt.savefig(f'{base}_{group_name}{ext}', bbox_inches='tight')

            results[group_name] = axes

        return results

    def _autocorrelation_plots(self, max_lag=None, figsize=(15, 10), per_chain=True):
        return self.autocorrelation_plots(max_lag=max_lag or 100, figsize=figsize, save_path=None)

    def _draw_g_prior(self, n_obs, n_mc):
        g_dist = getattr(self.sampler, 'g_dist', 'halfnormal')
        if g_dist == 'halfnormal':
            return np.abs(np.random.randn(n_mc, n_obs))
        elif g_dist == 'exponential':
            return np.random.exponential(scale=1.0, size=(n_mc, n_obs))
        else:
            raise ValueError(f"Unknown g_dist '{g_dist}'.")

    def predict_probabilities(self, X, use_mean_params=True, n_samples=1000, is_training_data=False,
                              training_indices=None, n_mc=300, prediction_mode="marginal"):
        X = np.asarray(X, dtype=float)
        combined = self.combined_samples

        include_skewness = getattr(self.sampler, "include_skewness", False)

        if prediction_mode not in {"conditional", "marginal"}:
            raise ValueError("prediction_mode must be either 'conditional' or 'marginal'.")

        if prediction_mode == "conditional":
            if not include_skewness:
                marginalize_g = False
            else:
                if not is_training_data:
                    raise ValueError(
                        "Conditional prediction requires training data because test g_i values are not sampled by the Gibbs sampler.")

                if "g" not in combined:
                    raise ValueError("Conditional prediction requested, but posterior g samples are not available.")

                if training_indices is None:
                    training_indices = np.arange(X.shape[0])

                training_indices = np.asarray(training_indices, dtype=int)
                marginalize_g = False

        else:
            marginalize_g = include_skewness

        if use_mean_params:
            beta_mean = np.mean(combined["beta"], axis=0)
            if "delta" in combined and include_skewness:
                delta_mean = np.mean(combined["delta"], axis=0)
            else:
                delta_mean = np.zeros(self.sampler.J_minus_1)
            if "Psi" in combined:
                Psi_use = np.mean(combined["Psi"], axis=0)
            else:
                Psi_use = self.sampler.Psi
            g_use = None

            if prediction_mode == "conditional" and include_skewness:
                g_mean = np.mean(combined["g"], axis=0)
                g_use = g_mean[training_indices]

            return self._compute_probabilities(X, beta_mean, delta_mean, g=g_use, Psi=Psi_use, n_mc=n_mc,
                                               marginalize_g=marginalize_g)

        n_obs = X.shape[0]

        beta_s = combined["beta"]
        n_total = beta_s.shape[0]

        if "delta" in combined and include_skewness:
            delta_s = combined["delta"]
        else:
            delta_s = np.zeros((n_total, self.sampler.J_minus_1))

        if "Psi" in combined:
            Psi_s = combined["Psi"]
        else:
            Psi_s = np.tile(self.sampler.Psi[None], (n_total, 1, 1))

        g_s = None

        if prediction_mode == "conditional" and include_skewness:
            g_s = combined["g"][:, training_indices]

        idx = np.random.choice(n_total, size=n_samples, replace=True)

        probs = np.zeros((n_samples, n_obs, self.sampler.J))

        for i, ix in enumerate(idx):
            if g_s is not None:
                g_i = g_s[ix]
            else:
                g_i = None

            probs[i] = self._compute_probabilities(X, beta_s[ix], delta_s[ix], g=g_i, Psi=Psi_s[ix], n_mc=n_mc,
                                                   marginalize_g=marginalize_g)

        out = np.clip(np.mean(probs, axis=0), 1e-15, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def _compute_probabilities(self, X, beta, delta, g=None, W=None, Psi=None, n_mc=300, marginalize_g=False):
        Psi = self.sampler.Psi if Psi is None else np.asarray(Psi)
        if W is not None:
            return self._mc_choice_probabilities_from_mean(np.asarray(W), Psi=Psi, n_mc=n_mc)
        include_skewness = getattr(self.sampler, "include_skewness", False)

        if not marginalize_g or not include_skewness:
            if include_skewness and not marginalize_g and g is None:
                raise ValueError(
                    "Conditional skewed prediction requires posterior g values. Use prediction_mode='marginal' if you want to integrate over new g_i.")

            U = self._compute_systematic_utilities(X, beta, delta, g=g)

            return self._mc_choice_probabilities_from_mean(U[:, 1:], Psi=Psi, n_mc=n_mc)

        # Test: marginalize over g
        X = np.asarray(X, dtype=float)
        n_obs = X.shape[0]
        mu_G = getattr(self.sampler, 'mu_G', np.sqrt(2.0 / np.pi))

        if beta.ndim == 1:
            beta_mat = beta.reshape((self.sampler.J_minus_1, self.sampler.p))
        else:
            beta_mat = beta
        base = np.dot(X, beta_mat.T)
        delta_arr = np.asarray(delta)
        g_draws = self._draw_g_prior(n_obs, n_mc)

        skew = (g_draws - mu_G)[:, :, None] * delta_arr[None, None, :]
        mean_diff = base[None, :, :] + skew

        n_eps = max(50, n_mc // 5)
        probs_sum = np.zeros((n_obs, self.sampler.J))
        for m in range(n_mc):
            p_m = self._mc_choice_probabilities_from_mean(mean_diff[m], Psi=Psi, n_mc=n_eps)
            probs_sum += p_m

        probs = probs_sum / n_mc
        probs = np.clip(probs, 1e-15, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict_choices(self, X, use_mean_params=True, is_training_data=False, training_indices=None, n_samples=300,
                        n_mc=50, prediction_mode="marginal"):
        probs = self.predict_probabilities(X, use_mean_params=use_mean_params, n_samples=n_samples,
                                           is_training_data=is_training_data, training_indices=training_indices,
                                           n_mc=n_mc, prediction_mode=prediction_mode)
        return np.argmax(probs, axis=1) + 1

    def _compute_systematic_utilities(self, X, beta, delta, g=None, return_utilities=False):
        X = np.asarray(X, dtype=float)
        mu_G = getattr(self.sampler, 'mu_G', np.sqrt(2.0 / np.pi))

        if beta.ndim == 1:
            beta = beta.reshape((self.sampler.J_minus_1, self.sampler.p))

        U = np.zeros((X.shape[0], self.sampler.J))
        base = np.dot(X, beta.T)
        if g is None:
            U[:, 1:] = base
        else:
            g = np.asarray(g, dtype=float)
            if g.ndim == 1:
                U[:, 1:] = base + (g - mu_G)[:, None] * np.asarray(delta)[None, :]
            else:
                U[:, 1:] = base + np.asarray(delta)[None, :] * (g - mu_G)
        return U

    def _mc_choice_probabilities_from_mean(self, mean_diff, Psi=None, n_mc=300, rng=None):
        mean_diff = np.asarray(mean_diff, dtype=float)
        if mean_diff.ndim != 2:
            raise ValueError("mean_diff must be a 2D array with shape (n_obs, J_minus_1).")
        Psi = (self.sampler.Psi if Psi is None else np.asarray(Psi, dtype=float))

        rng = np.random.default_rng() if rng is None else rng
        n_obs, j_minus_1 = mean_diff.shape
        if Psi.shape != (j_minus_1, j_minus_1):
            raise ValueError(f"Psi must have shape ({j_minus_1}, {j_minus_1}), but got {Psi.shape}.")
        probs = np.zeros((n_obs, j_minus_1 + 1), dtype=float)

        chunk = min(200, n_mc)
        done = 0
        while done < n_mc:
            m = min(chunk, n_mc - done)
            eps = rng.multivariate_normal(mean=np.zeros(j_minus_1), cov=Psi, size=m)
            W = mean_diff[None, :, :] + eps[:, None, :]
            ref = np.all(W <= 0, axis=2)
            probs[:, 0] += ref.sum(axis=0)

            nonref = ~ref

            if np.any(nonref):
                winners = np.argmax(W, axis=2) + 1
                for cls in range(1, j_minus_1 + 1):
                    probs[:, cls] += np.sum(nonref & (winners == cls), axis=0)
            done += m
        probs /= float(n_mc)
        probs = np.clip(probs, 1e-15, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def evaluate_prediction_performance(self, X, y, data_type, prediction_mode, label=None, samples_list=None,
                                        include_waic=False, n_samples=300, n_mc=50, training_indices=None):
        if label is None:
            label = f"{data_type}_{prediction_mode}"

        if data_type == "train" and training_indices is None:
            training_indices = np.arange(len(y))

        y_probs = self.predict_probabilities(X, use_mean_params=False, n_samples=n_samples,
                                             is_training_data=(data_type == "train"), training_indices=training_indices,
                                             n_mc=n_mc, prediction_mode=prediction_mode)
        y_pred = np.argmax(y_probs, axis=1) + 1
        metrics = self._compute_all_metrics(y, y_pred, y_probs, data_type=data_type, samples_list=samples_list,
                                            X=X if data_type == "train" else None, include_waic=include_waic)

        metrics["prediction_mode"] = prediction_mode
        metrics["evaluation_label"] = label

        return metrics

    def evaluate_train_performance(self, X_train=None, y_train=None, samples_list=None, include_waic=False,
                                   prediction_mode="conditional", label=None, n_samples=300, n_mc=50):
        if X_train is None:
            X_train = self.sampler.X

        if y_train is None:
            y_train = self.y_true

        idx = np.arange(len(y_train))

        if label is None:
            label = f"train_{prediction_mode}"

        return self.evaluate_prediction_performance(X=X_train, y=y_train, data_type="train",
                                                    prediction_mode=prediction_mode, label=label,
                                                    samples_list=samples_list, include_waic=include_waic,
                                                    n_samples=n_samples, n_mc=n_mc, training_indices=idx)

    def evaluate_test_performance(self, X_test=None, y_test=None, samples_list=None, prediction_mode="marginal",
                                  label=None, n_samples=300, n_mc=50):
        if X_test is None:
            X_test = self.X_test

        if y_test is None:
            y_test = self.y_test

        if prediction_mode != "marginal":
            raise ValueError("Test performance should use prediction_mode='marginal'.")

        if label is None:
            label = "test_marginal"

        return self.evaluate_prediction_performance(X=X_test, y=y_test, data_type="test", prediction_mode="marginal",
                                                    label=label, samples_list=samples_list, include_waic=False,
                                                    n_samples=n_samples, n_mc=n_mc, training_indices=None)

    def _compute_all_metrics(self, y_true, y_pred, y_probs, data_type='train', samples_list=None, X=None,
                             include_waic=False):
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except ValueError:
            auc = np.nan

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        class_labels = list(range(1, self.sampler.J + 1))
        recall_per_class = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        log_score = self._compute_log_score(y_true, y_probs)
        brier_score = self._compute_brier_score(y_true, y_probs)
        mae_choices = self._compute_mae_choices(y_true, y_pred)
        ks_stat = self._compute_kolmogorov_smirnov(y_true, y_probs)

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': prec,
            'recall_macro': rec,
            'log_score': log_score,
            'brier_score': brier_score,
            'MAE_choices': mae_choices,
            'KS_statistic': ks_stat,
            **{f'recall_class_{c}': float(recall_per_class[k])
               for k, c in enumerate(class_labels)},
            **{f'precision_class_{c}': float(precision_per_class[k])
               for k, c in enumerate(class_labels)},
            **{f'f1_class_{c}': float(f1_per_class[k])
               for k, c in enumerate(class_labels)},
            'data_type': data_type
        }

        if data_type == 'train':
            metrics.update({'AIC_pseudo': None, 'BIC_pseudo': None})
            if samples_list is not None and X is not None:
                try:
                    metrics.update(self._compute_dic(y_true, X))
                except Exception:
                    metrics.update({'DIC': None, 'pD': None})
            else:
                metrics.update({'DIC': None, 'pD': None})

            if include_waic:
                try:
                    metrics.update(self._compute_waic())
                except Exception as e:
                    print(f"WAIC failed: {e}")
                    metrics['waic'] = np.nan
        else:
            metrics.update({'AIC_pseudo': None, 'BIC_pseudo': None, 'DIC': None, 'pD': None, 'waic': None})

        return metrics

    def _compute_log_score(self, y_true, y_probs):
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        ls = sum(np.log(np.clip(y_probs[i, y_true[i] - 1], 1e-15, 1.0)) for i in range(len(y_true)))
        return -ls / len(y_true)

    def _compute_brier_score(self, y_true, y_probs):
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        n, nc = len(y_true), y_probs.shape[1]
        oh = np.zeros((n, nc))
        for i in range(n):
            oh[i, y_true[i] - 1] = 1.0
        return float(np.mean(np.sum((y_probs - oh) ** 2, axis=1)))

    def _compute_mae_choices(self, y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def _compute_kolmogorov_smirnov(self, y_true, y_probs):
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        nc = y_probs.shape[1]
        obs = np.array([np.mean(y_true == i + 1) for i in range(nc)])
        prd = np.mean(y_probs, axis=0)
        return float(np.max(np.abs(np.cumsum(obs) - np.cumsum(prd))))

    def _compute_dic(self, y_true, X, max_draws=200, n_mc=200):
        combined = self.combined_samples
        n_total = combined['beta'].shape[0]
        draw_idx = np.linspace(0, n_total - 1, min(max_draws, n_total), dtype=int)
        deviances = []
        for idx in draw_idx:
            beta = combined['beta'][idx]
            delta = (combined['delta'][idx] if (
                    'delta' in combined and getattr(self.sampler, 'include_skewness', False)) else np.zeros(
                self.sampler.J_minus_1))
            Psi = combined['Psi'][idx] if 'Psi' in combined else self.sampler.Psi
            g = (combined['g'][idx] if ('g' in combined and getattr(self.sampler, 'include_skewness', False)) else None)
            probs = self._compute_probabilities(X, beta, delta, g=g, Psi=Psi, n_mc=n_mc)
            deviances.append(2.0 * len(y_true) * self._compute_log_score(y_true, probs))

        mean_deviance = float(np.mean(deviances))
        beta_mean = np.mean(combined['beta'], axis=0)
        delta_mean = (np.mean(combined['delta'], axis=0)
                      if ('delta' in combined and getattr(self.sampler, 'include_skewness', False)) else np.zeros(
            self.sampler.J_minus_1))
        Psi_mean = (np.mean(combined['Psi'], axis=0) if 'Psi' in combined else self.sampler.Psi)
        g_mean = (np.mean(combined['g'], axis=0) if (
                'g' in combined and getattr(self.sampler, 'include_skewness', False)) else None)
        probs_mean = self._compute_probabilities(X, beta_mean, delta_mean, g=g_mean, Psi=Psi_mean, n_mc=n_mc)
        deviance_at_mean = 2.0 * len(y_true) * self._compute_log_score(y_true, probs_mean)
        p_dic = mean_deviance - deviance_at_mean
        return {'DIC': mean_deviance + p_dic, 'pD': p_dic, 'p_DIC': p_dic, 'mean_deviance': mean_deviance,
                'deviance_at_mean': deviance_at_mean}

    def _compute_waic(self, max_draws=200, n_mc=200):
        combined = self.combined_samples
        n_total = combined['beta'].shape[0]
        draw_idx = np.linspace(0, n_total - 1, min(max_draws, n_total), dtype=int)
        y_true = np.asarray(self.y_true)
        X = np.asarray(self.sampler.X, dtype=float)
        S, N = len(draw_idx), len(y_true)
        loglik = np.zeros((S, N))
        for s, idx in enumerate(draw_idx):
            beta = combined['beta'][idx]
            delta = (combined['delta'][idx] if (
                    'delta' in combined and getattr(self.sampler, 'include_skewness', False)) else np.zeros(
                self.sampler.J_minus_1))
            Psi = combined['Psi'][idx] if 'Psi' in combined else self.sampler.Psi
            g = (combined['g'][idx] if ('g' in combined and getattr(self.sampler, 'include_skewness', False)) else None)
            probs = self._compute_probabilities(X, beta, delta, g=g, Psi=Psi, n_mc=n_mc)
            loglik[s] = np.log(np.clip(probs[np.arange(N), y_true - 1], 1e-15, 1.0))

        lppd = float(np.sum(np.log(np.mean(np.exp(loglik), axis=0))))
        p_waic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
        return {'waic': -2.0 * (lppd - p_waic), 'p_waic': p_waic, 'lppd': lppd}

    def compute_waic(self):
        return self._compute_waic()

    def confusion_matrix_plot(self, X=None, y=None, figsize=(8, 6), data='train', prediction_mode="marginal",
                              use_mean_params=False, n_samples=300, n_mc=50):
        if data == 'train':
            if X is None:
                X = self.sampler.X
            if y is None:
                y = self.y_true

            training_indices = np.arange(len(y))
            is_training_data = True

        else:
            if X is None:
                X = self.X_test
            if y is None:
                y = self.y_test

            if prediction_mode != "marginal":
                raise ValueError("Test confusion matrix should use prediction_mode='marginal'.")

            training_indices = None
            is_training_data = False

        y_pred = self.predict_choices(X, use_mean_params=use_mean_params, is_training_data=is_training_data,
                                      training_indices=training_indices, n_samples=n_samples, n_mc=n_mc,
                                      prediction_mode=prediction_mode)

        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, self.sampler.J + 1),
                    yticklabels=range(1, self.sampler.J + 1))

        plt.title(f'Confusion Matrix: {data}_{prediction_mode}')
        plt.xlabel('Predicted Choice')
        plt.ylabel('True Choice')

        return cm

    def classification_report(self, X=None, y=None, data='train', prediction_mode="marginal", use_mean_params=False,
                              n_samples=300, n_mc=50):
        if data == 'train':
            if X is None:
                X = self.sampler.X
            if y is None:
                y = self.y_true

            training_indices = np.arange(len(y))
            is_training_data = True

        else:
            if X is None:
                X = self.X_test
            if y is None:
                y = self.y_test

            if prediction_mode != "marginal":
                raise ValueError("Test classification report should use prediction_mode='marginal'.")
            training_indices = None
            is_training_data = False

        y_pred = self.predict_choices(X, use_mean_params=use_mean_params, is_training_data=is_training_data,
                                      training_indices=training_indices, n_samples=n_samples, n_mc=n_mc,
                                      prediction_mode=prediction_mode)

        return classification_report(y, y_pred, labels=list(range(1, self.sampler.J + 1)),
                                     target_names=[f'Alternative {i}' for i in range(1, self.sampler.J + 1)],
                                     zero_division=0)
