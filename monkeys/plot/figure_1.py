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
from analysis.sigmoid_fit.sigmoid_fit import sigmoid

from plot.tools.tools import add_letter, add_text

from parameters.parameters import CONTROL_CONDITIONS, \
    FIG_FOLDER, GAIN, LOSS


def _line(x, risk_aversion, class_model, ax,
          alpha=1.0,
          linewidth=3, color="C0", linestyle="-"):

    y = [class_model.u(x=i, risk_aversion=risk_aversion) for i in x]

    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
            linestyle=linestyle)


def plot(ax, data, linestyles=None, color='C0', alpha_chunk=0.5):
    """
    Produce the utility function figure
    """

    pr = data['risk_aversion']
    class_model = data['class_model']
    cond = data['cond']

    if linestyles is None:
        linestyles = ['-' for _ in range(len(pr))]

    if cond == GAIN:
        x = np.linspace(0, 1, 1000)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.plot((0, 1), (0, 1), alpha=0.5, linewidth=1, color='black',
                linestyle='--', zorder=-10)
    elif cond == LOSS:
        x = np.linspace(-1, 0, 1000)
        ax.set_xlim(-1, 0)
        ax.set_ylim(-1, 0)

        ax.set_xticks([-1, 0])
        ax.set_yticks([-1, 0])

        ax.plot((-1, 0), (-1, 0), alpha=0.5, linewidth=1, color='black',
                linestyle='--', zorder=-10)
    else:
        raise ValueError

    for j in range(len(pr)):
        _line(
            x=x,
            class_model=class_model,
            risk_aversion=pr[j],
            color=color,
            ax=ax, linewidth=1, alpha=alpha_chunk,
            linestyle=linestyles[j]
        )

    v_mean = np.mean(pr)
    v_std = np.std(pr)
    _line(
        x=x,
        risk_aversion=v_mean,
        class_model=class_model,
        ax=ax,
        color=color
    )

    add_text(ax, r'$\omega=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$')

    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')

    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")

    ax.set_aspect(1)


def figure_1(a, alpha_chunk=0.5):

    nrows, ncols = 2, len(CONTROL_CONDITIONS) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4*ncols, 4*nrows))

    colors = {GAIN: 'C0', LOSS: 'C1'}

    linestyles = {m: '--' if m in ('Havane', 'Gladys') else ':'
                  for m in a.monkeys}

    idx_subplot = 0

    for i, cond in enumerate((GAIN, LOSS)):

        for j, control_condition in \
                enumerate(CONTROL_CONDITIONS + ('freq_risk_data', )):

            ax = axes[i, j]

            # sig_mid = np.zeros(n_monkey)
            # sig_steep = np.zeros(n_monkey)
            # min_x, max_x = 0, 0

            x_fit_all = []
            y_fit_all = []

            for k, m in enumerate(a.monkeys):
                if control_condition in CONTROL_CONDITIONS:
                    d = a.control_sigmoid_data[cond][m][control_condition]
                else:
                    d = a.freq_risk_data[cond][m]

                x_fit = d['fit']['x']
                y_fit = d['fit']['y']

                x_fit_all.append(x_fit)
                y_fit_all.append(y_fit)

                # sig_mid[k] = d['fit']['sig_mid']
                # sig_steep[k] = d['fit']['sig_steep']
                # min_x, max_x = np.min(x_fit), np.max(x_fit)

                ax.plot(
                    x_fit, y_fit, linewidth=1, alpha=alpha_chunk,
                    linestyle=linestyles[m], color=colors[cond])

            # mean_sig_mid = np.mean(sig_mid)
            # std_sig_mid = np.std(sig_mid)
            # mean_sig_steep = np.mean(sig_steep)
            # std_sig_steep = np.std(sig_steep)

            #p_opt = mean_sig_mid, mean_sig_steep
            # x_fit = np.linspace(min_x, max_x, n_points)
            # y_fit = sigmoid(x_fit, *p_opt)

            # ax.plot(x_fit, y_fit,  linewidth=3, alpha=1.0,
            #         linestyle="-", color=colors[cond])

            ax.plot(np.mean(np.asarray(x_fit_all), axis=0),
                    np.mean(np.asarray(y_fit_all), axis=0), linewidth=3, alpha=1.0,
                    linestyle="-", color=colors[cond])

            # add_text(ax,
            #          r'$x_0=' + f'{mean_sig_mid:.2f}\pm{std_sig_mid:.2f}' + '$' + '\n' +
            #          r'$k=' + f'{mean_sig_steep:.2f}\pm{std_sig_steep:.2f}' + '$')


            ax.spines['right'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['top'].set_color('none')

            x_label = "$EV_{right} - EV_{left}$"
            if control_condition in CONTROL_CONDITIONS:
                y_label = "p(choose right)"
            else:
                y_label = "p(choose riskiest)"

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
                       linestyle='--', zorder=-10)
            ax.axvline(0.0, alpha=0.5, linewidth=1, color='black',
                       linestyle='--', zorder=-10)

            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])

            add_letter(axes[i, j], i=idx_subplot)

            idx_subplot += 1

    fig_path = os.path.join(FIG_FOLDER, f"figure_1.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"Figure {fig_path} created!")