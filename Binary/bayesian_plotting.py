"""
This file generates the trace and ppc plots.
"""

import arviz as az
import numpy as np


def nonskewed_trace(X):
    axes = az.plot_trace(X, var_names=['betas'], show=False)
    axes[0, 0].set_ylabel("Density")
    axes[0, 1].set_ylabel("Sample Values")
    axes[0, 0].set_xlabel("Sample Values")
    axes[0, 1].set_xlabel("Draw Index")

    return

def skewed_trace(X):
    axes = az.plot_trace(X, var_names=['betas', 'delta'], show=False)
    axes[1, 0].set_xlabel("Sample Values")
    axes[1, 1].set_xlabel("Draw Index")
    axes[0, 0].set_ylabel("Density")
    axes[0, 1].set_ylabel("Sample Values")
    axes[1, 0].set_ylabel("Density")
    axes[1, 1].set_ylabel("Sample Values")

    return


def nonskewed_ppc(X):
    ax = az.plot_ppc(X, group='posterior', kind='kde', figsize=(6, 5))

    #ax.set_ylim(0, 0.7)
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_ylabel("Probability", fontsize=16)
    ax.set_xlabel("Outcome", fontsize=16)
    # ax.set_xticks([0.5, 1.5, 2.5])
    # ax.set_xticklabels(class_names, fontsize=12)
    # Increase the size of the legend
    ax.legend(fontsize=12)
    ax.grid(False)

    return


def skewed_ppc(X):
    ax = az.plot_ppc(X, group='posterior', kind='kde', figsize=(6, 5))

    #ax.set_ylim(0, 0.7)
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_ylabel("Probability", fontsize=16)
    ax.set_xlabel("Outcome", fontsize=16)
    # ax.set_xticks([0.5, 1.5, 2.5])
    # ax.set_xticklabels(class_names, fontsize=12)
    # Increase the size of the legend
    ax.legend(fontsize=12)
    ax.grid(False)

    return
