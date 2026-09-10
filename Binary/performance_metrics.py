"""
This file is to evaluate the performance of the models.
"""

import arviz as az
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, brier_score_loss, log_loss,
                              confusion_matrix, average_precision_score)


def _loo_stats(idata):
    if idata is None:
        return None, None, None, None
    loo = az.loo(idata)
    elpd_loo = getattr(loo, 'elpd_loo', None)
    if elpd_loo is None:
        elpd_loo = loo.elpd
    elpd_loo_se = loo.se
    loo_value, loo_se = -2 * elpd_loo, elpd_loo_se * 2
    return elpd_loo, elpd_loo_se, loo_value, loo_se


def training_perf_metrics(ytrain, ypred_train, train_probs, idata=None):
    train_acc = accuracy_score(ytrain, ypred_train)
    train_prec = precision_score(ytrain, ypred_train, zero_division=0)
    train_rec = recall_score(ytrain, ypred_train, zero_division=0)
    train_f1 = f1_score(ytrain, ypred_train, zero_division=0)
    train_auc = roc_auc_score(ytrain, train_probs)
    train_brier = brier_score_loss(ytrain, train_probs)
    train_log_loss = log_loss(ytrain, train_probs, labels=[0, 1])
    train_auprc = average_precision_score(ytrain, train_probs)
    train_cm = (str(confusion_matrix(ytrain, ypred_train).flatten(order='C')))[1:-1]

    elpd_loo, elpd_loo_se, loo_value, loo_se = _loo_stats(idata)

    return (train_acc, train_prec, train_rec, train_f1, train_auc, train_log_loss, train_brier, train_auprc,
            elpd_loo, elpd_loo_se, loo_value, loo_se, train_cm)


def testing_perf_metrics(ytest, ypred_test, test_probs, idata=None):
    test_acc = accuracy_score(ytest, ypred_test)
    test_prec = precision_score(ytest, ypred_test, zero_division=0)
    test_rec = recall_score(ytest, ypred_test, zero_division=0)
    test_f1 = f1_score(ytest, ypred_test, zero_division=0)
    test_log_loss = log_loss(ytest, test_probs, labels=[0, 1])
    test_auc = roc_auc_score(ytest, test_probs)
    test_brier = brier_score_loss(ytest, test_probs)
    test_auprc = average_precision_score(ytest, test_probs)
    test_cm = (str(confusion_matrix(ytest, ypred_test).flatten(order='C')))[1:-1]

    elpd_loo, elpd_loo_se, loo_value, loo_se = _loo_stats(idata)

    return (test_acc, test_prec, test_rec, test_f1, test_auc, test_log_loss, test_brier, test_auprc,
            elpd_loo, elpd_loo_se, loo_value, loo_se, test_cm)


def recovery_summary(trace, true_beta, model_type):
    var_names = ['betas'] + (['delta'] if model_type == 'skewed' else [])
    summ = az.summary(trace, var_names=var_names, kind='stats', round_to='none')

    result = {}
    for j in range(len(true_beta)):
        result[f'beta{j}_true'] = true_beta[j]
        result[f'beta{j}_postmean'] = summ.loc[f'betas[x{j}]', 'mean']
        result[f'beta{j}_postsd'] = summ.loc[f'betas[x{j}]', 'sd']
    if model_type == 'skewed':
        result['delta_postmean'] = summ.loc['delta', 'mean']
        result['delta_postsd'] = summ.loc['delta', 'sd']
    return result
