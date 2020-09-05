import os
import numpy as np
import matplotlib.pyplot as plt

from .subplot import precision
from .subplot import probability_distortion
from .subplot import utility

from plot.tools.tools import add_letter

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
        for param in ("risk_aversion", "distortion",
                      "precision", "side_bias"):
            data[param] = [np.mean(a.cpt_fit[cond][m][param])
                           for m in a.monkeys]

        utility.plot(ax=axes[i, 0], data=data, color=colors[i],
                     linestyles=linestyles)
        add_letter(axes[i, 0], i=i*3)
        probability_distortion.plot(
            ax=axes[i, 1], data=data,
            color=colors[i],
            linestyles=linestyles)
        add_letter(axes[i, 1], i=i*3+1)
        precision.plot(ax=axes[i, 2], data=data, color=colors[i],
                       linestyles=linestyles)
        add_letter(axes[i, 2], i=i * 3 + 2)

    fig_path = os.path.join(FIG_FOLDER, "figure_2.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Figure {fig_path} created!")
