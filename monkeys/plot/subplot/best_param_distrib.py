import numpy as np
import matplotlib.pyplot as plt

from plot.tools.tools import scatter_boxplot


def plot(fit, param_labels, fig_path):

    n_param = len(param_labels)

    # Colors
    colors = [f"C{i}" for i in range(n_param)]

    fig, axes = plt.subplots(nrows=n_param, figsize=(4, 3*n_param))

    for i in range(n_param):

        ax = axes[i]
        param_name = param_labels[i]
        title = f"Distribution of best-fit values for {param_name}"
        data = [np.mean(fit[k][param_name]) for k in fit.keys()]
        err = [np.std(fit[k][param_name]) for k in fit.keys()]

        scatter_boxplot(ax=ax, data=data,
                        err=err,
                        title=title,
                        y_label="Value",
                        x_tick_label=param_name,
                        color=colors[i],
                        dot_size=40)
    plt.tight_layout()
    plt.savefig(fig_path)


def demo():
    fit = {"mike": {"alpha": np.random.random(10), "epsilon": np.random.random(10)},
           "brian": {"alpha": np.random.random(10), "epsilon": np.random.random(10)},}
    plot(fit=fit)


if __name__ == "__main__":
    demo()
