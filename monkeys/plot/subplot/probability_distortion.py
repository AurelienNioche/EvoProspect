import numpy as np

from plot.tools.tools import add_text


# def pi(p, distortion):
#     """Probability distortion"""
#
#     return np.exp(-(-np.log(p)) ** distortion) if p > 0 else 0


def _line(param, ax,
          class_model,
          linestyle='-',
          linewidth=3, alpha=1.0, color='C0', n_points=1000):

    x = np.linspace(0, 1, n_points)

    ax.plot(x, [class_model.pi(i, param) for i in x], color=color,
            linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def plot(ax, data, linestyles=None, color='C0', label_font_size=20, ticks_label_size=14):
    """
    Produce the probability distortion figure
    """

    fit_distortion = data["distortion"]
    class_model = data["class_model"]

    if linestyles is None:
        linestyles = ['-' for _ in range(len(fit_distortion))]

    for j in range(len(fit_distortion)):

        _line(
            class_model=class_model,
            param=fit_distortion[j],
            color=color,
            alpha=0.5,
            linewidth=1,
            linestyle=linestyles[j],
            ax=ax)

    v_mean = np.mean(fit_distortion)
    v_std = np.std(fit_distortion)
    _line(
        class_model=class_model,
        param=v_mean,
        linewidth=3,
        color=color,
        ax=ax)
    add_text(ax, r'$\alpha=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$')

    ax.set_xlabel('$p$', fontsize=label_font_size)
    ax.set_ylabel('$w(p)$', fontsize=label_font_size)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_aspect('equal')

    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')

    ax.plot((0, 1), (0, 1), alpha=0.5, linewidth=1, color='black',
            linestyle='--', zorder=-10)

    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks((0, 0.5, 1))

    ax.tick_params(axis='both', which='major', labelsize=ticks_label_size)
