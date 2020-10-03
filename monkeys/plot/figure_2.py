import os
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_lowercase

from .subplot import precision
from .subplot import probability_distortion
from .subplot import utility

from parameters.parameters import FIG_FOLDER, GAIN, LOSS


def figure_2(a):

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4*ncols, 4*nrows))

    colors = ['C0', 'C1']

    linestyles = ['--' if i in ('Havane', 'Gladys') else ':'
                  for i in a.monkeys]

    for i, cond in enumerate((GAIN, LOSS)):

        data = {'class_model': a.class_model,
                'cond': cond}
        for param in ("risk_aversion", "distortion", "precision", "side_bias"):
            data[param] = [np.mean(a.cpt_fit[cond][m][param])
                           for m in a.monkeys]

        ax = axes[i, 0]
        utility.plot(ax=ax, data=data, color=colors[i],
                     linestyles=linestyles)
        ax.text(-0.1, 1.1, ascii_lowercase[i*3],
                transform=ax.transAxes, size=20, weight='bold')
        ax.set_title(f"Utility function\n({cond})",
                     size=11, weight="bold")

        ax = axes[i, 1]
        probability_distortion.plot(
            ax=axes[i, 1], data=data,
            color=colors[i],
            linestyles=linestyles)
        ax.text(-0.1, 1.1, ascii_lowercase[i*3+1],
                transform=ax.transAxes, size=20, weight='bold')
        ax.set_title(f"Probability weighting function\n({cond})",
                     size=11, weight="bold")

        ax = axes[i, 2]
        precision.plot(ax=ax, data=data, color=colors[i],
                       linestyles=linestyles)
        ax.text(-0.1, 1.1, ascii_lowercase[i*3+2],
                transform=ax.transAxes, size=20, weight='bold')
        ax.set_title(f"Softmax function\n({cond})",
                     size=11, weight='bold')

    plt.tight_layout()

    fig_path = os.path.join(FIG_FOLDER, "figure_2.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure {fig_path} created!")

    fig_path = os.path.join(FIG_FOLDER, f"figure_2.pdf")
    plt.savefig(fig_path)
    print(f"Figure {fig_path} created!")
