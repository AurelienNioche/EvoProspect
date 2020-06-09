import matplotlib.pyplot as plt
import numpy as np

from plot.tools.tools import scatter_boxplot


def plot(fit, fig_path_lls, fig_path_bic):

    for score, fig_path in zip(('LLS', 'BIC'),
                               (fig_path_lls, fig_path_bic)):

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

