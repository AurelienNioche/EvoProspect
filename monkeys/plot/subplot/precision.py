import numpy as np

from plot.tools.tools import add_text


def plot(ax, data, linestyles=None, color='C0'):
    """
    Produce the precision figure
    """

    n_points = 1000

    v_mean = np.mean(data['precision'])
    v_std = np.std(data['precision'])

    n_chunk = len(data['precision'])

    class_model = data['class_model']

    if class_model.__name__ == "DMSciReports":

        p0 = 0.5
        p1, x1 = 1., 0.25

        x0_equal_ev = x1 * (1 / p0)

        x0_list = np.linspace(x1 + 0.01, 1.00, n_points)

        x = x0_list / x1

        pairs = []

        for i, x0 in enumerate(x0_list):
            pairs.append({"p0": p0, "x0": x0, "p1": p1, "x1": x1})

        y = np.zeros((n_chunk, len(x)))

        for i_c in range(n_chunk):

            dm = class_model([data[k][i_c] for k in class_model.param_labels])

            for i_p, p in enumerate(pairs):
                y[i_c, i_p] = dm.p_choice(c=0, **p)

        dm = class_model([np.mean(data[k]) for k in class_model.param_labels])
        y_mean = np.zeros(len(x))
        for i_p, p in enumerate(pairs):
            y_mean[i_p] = dm.p_choice(c=0, **p)

        ax.axvline(x0_equal_ev / x1, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)

        x_label = r"$\frac{x_{risky}}{x_{safe}}$"
        y_label = "P(Choose risky option)"
        text = r'$\lambda=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$'

    elif class_model.__name__ == "AgentSoftmax":

        fit_precision = data['precision']

        x = np.linspace(-1, 1, n_points)
        y = np.zeros((n_chunk, len(x)))
        for i_c in range(n_chunk):
            v = fit_precision[i_c]
            y[i_c] = class_model.softmax(x, v)

        y_mean = np.zeros(len(x))
        y_mean[:] = class_model.softmax(x, v_mean)

        ax.axvline(0, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)

        x_label = r"$SEU(L_{1}) - SEU(L_{2})$"
        y_label = "$P(Choose L_{1})$"
        text = r'$\lambda=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$'

    elif class_model.__name__ == "AgentSideAdditive":

        fit_precision = data['precision']
        fit_side_bias = data['side_bias']

        x = np.linspace(-1, 1, n_points)
        y = np.zeros((n_chunk, len(x)))
        for i_c in range(n_chunk):
            v = fit_precision[i_c]
            x_biased = x + fit_side_bias[i_c]
            y[i_c] = class_model.softmax(x_biased, v)

        y_mean = np.zeros(len(x))
        mean_side_bias = np.mean(fit_side_bias)
        std_side_bias = np.std(fit_side_bias)
        x_biased = x + mean_side_bias
        y_mean[:] = class_model.softmax(x_biased, v_mean)

        ax.axvline(0, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)

        x_label = r"$SEU(L_{right}) - SEU(L_{left})$"
        y_label = "$P(Choose\,L_{right})$"
        text = r'$\lambda=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$\n' \
            + r'$\gamma=' + f'{mean_side_bias:.2f}\pm{std_side_bias:.2f}' + '$\n'

    elif class_model.__name__ == "AgentSide":

        fit_precision = data['precision']
        # fit_side_bias = data['side_bias']

        x = np.linspace(-1, 1, n_points)
        y = np.zeros((n_chunk, len(x)))
        for i_c in range(n_chunk):
            v = fit_precision[i_c]
            y[i_c] = class_model.softmax(x, v)

        y_mean = np.zeros(len(x))
        y_mean[:] = class_model.softmax(x, v_mean)

        ax.axvline(0, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)

        x_label = r"$SEU(L_{right}) - SEU(L_{left})$"
        y_label = "$p(choose L_{right})$"
        text = r'$\lambda=' + f'{v_mean:.2f}\pm{v_std:.2f}' + '$'

    else:
        raise ValueError

    if linestyles is None:
        linestyles = ['-' for _ in range(len(data['precision']))]

    for i_c in range(n_chunk):
        ax.plot(x, y[i_c], color=color, linewidth=1, alpha=0.5,
                linestyle=linestyles[i_c])

    # show_average
    ax.plot(x, y_mean, color=color, linewidth=3, alpha=1)

    add_text(ax, text)

    # ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])

    # ax.set_xlim(0.0, 2.01)
    ax.set_ylim(-0.01, 1.01)

    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')

    ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
