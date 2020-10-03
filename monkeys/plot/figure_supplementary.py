import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .subplot import history_control
from .subplot import precision
from .subplot import probability_distortion
from .subplot import utility
from .subplot import control
from .subplot import freq_risk
from .subplot import control_sigmoid
from .subplot import info
from .subplot import best_param_distrib
from .subplot import LLS_BIC_distrib

from plot.tools.tools import add_letter

from parameters.parameters import CONTROL_CONDITIONS, \
    FIG_FOLDER, GAIN, LOSS


def figure_supplementary(a):

    print("Monkeys list", a.monkeys)
    fig_ind(a)
    fig_control(a)
    fig_freq_risk(a)


def fig_control(a):

    for control_condition in CONTROL_CONDITIONS:
        nrows, ncols = 3, 6
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(2.5*ncols, 2.5*nrows))

        axes = axes.flatten()

        colors = ['C0', 'C1']
        show_ylabel = [True, False]

        k = 0
        for i, m in enumerate(a.monkeys):
            add_letter(axes[k], i=i)
            for j, cond in enumerate((GAIN, LOSS)):
                control_sigmoid.plot(
                    ax=axes[k], data=a.control_sigmoid_data[cond][m],
                    control_condition=control_condition,
                    color=colors[j], show_ylabel=show_ylabel[j],
                    dot_size=50)
                k += 1

        fig_path = os.path.join(FIG_FOLDER, f"SUP_{control_condition}.pdf")
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Figure {fig_path} created!")


def fig_freq_risk(a):

    nrows, ncols = 3, 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(2.5*ncols, 2.5*nrows))

    axes = axes.flatten()

    colors = ['C0', 'C1']
    show_ylabel = [True, False]

    k = 0
    for i, m in enumerate(a.monkeys):
        add_letter(axes[k], i=i)
        for j, cond in enumerate((GAIN, LOSS)):
            freq_risk.plot(ax=axes[k], data=a.freq_risk_data[cond][m],
                           color=colors[j], show_ylabel=show_ylabel[j],
                           dot_size=50)
            k += 1

    fig_path = os.path.join(FIG_FOLDER, "SUP_freq_risk.pdf")
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Figure {fig_path} created!")


def fig_ind(a):

    colors = ['C0', 'C1']
    # show_ylabel = [True, False]


    for j, cond in enumerate((GAIN, LOSS)):
        nrows, ncols = 5, 6
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(2.5 * ncols, 2.5 * nrows))

        axes = axes.flatten()
        k = 0
        for i, m in enumerate(a.monkeys):
            add_letter(axes[k], i=i)
            data = {'class_model': a.class_model,
                    'cond': cond}
            for param in ("risk_aversion", "distortion",
                          "precision", "side_bias"):
                data[param] = a.cpt_fit[cond][m][param]

            utility.plot(ax=axes[k], data=data, color=colors[j])
            probability_distortion.plot(
                ax=axes[k+1], data=data, color=colors[j])
            precision.plot(ax=axes[k+2], data=data, color=colors[j])
            k += 3

        for ax in axes[k:]:
            ax.set_axis_off()

        fig_path = os.path.join(FIG_FOLDER, f"SUP_ind_{cond}.pdf")
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Figure {fig_path} created!")
