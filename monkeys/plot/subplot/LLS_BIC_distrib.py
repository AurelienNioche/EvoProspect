import matplotlib.pyplot as plt
import numpy as np

from plot.tools.tools import scatter_boxplot


def _plot(fit, score, fig_path):
    fig, ax = plt.subplots(figsize=(4, 3))

    title = f"Distribution of {score}"
    data = [np.mean(fit[k][score]) for k in fit.keys()]
    err = [np.std(fit[k][score]) for k in fit.keys()]

    scatter_boxplot(ax=ax, data=data,
                    err=err,
                    title=title,
                    y_label="Value",
                    x_tick_label=score,
                    color='C0',
                    dot_size=40)
    plt.tight_layout()
    plt.savefig(fig_path)


def plot_bic(fit, fig_path):

    _plot(fit=fit, score='BIC', fig_path=fig_path)


def plot_LLS(fit, fig_path):

    _plot(fit=fit, score='LLS', fig_path=fig_path)



