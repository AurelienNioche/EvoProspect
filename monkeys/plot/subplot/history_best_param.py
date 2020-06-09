import numpy as np
import warnings


def _plot_history_best_param(
        param, ax, y_lim, param_name, mid_line=None,
        axis_label_font_size=20,
        ticks_label_font_size=14,
        color='C0',
        point_size=100):

    x_data = np.arange(len(param)) + 1
    y_data = param

    ax.scatter(x_data, y_data, color=color, alpha=0.5, s=point_size)

    if mid_line is not None:
        ax.axhline(mid_line, alpha=0.5, linewidth=1, color='black',
                   linestyle='--',
                   zorder=-10)

    ax.set_xlim((0.5, len(x_data)+0.5))
    ax.set_ylim(y_lim)

    if len(x_data) >= 10:
        ax.set_xticks((1, len(x_data)//2, len(x_data)))

    # Axis labels
    ax.set_xlabel(
        "time",
        fontsize=axis_label_font_size)
    ax.set_ylabel(
        param_name,
        fontsize=axis_label_font_size)

    # Remove top and right borders
    # ax.spines['right'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['top'].set_color('none')

    ax.tick_params(axis='both', which='major', labelsize=ticks_label_font_size)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_label_font_size)


def plot(axes, data):

    fit = data['fit']
    class_model = data['fit']['class_model']

    args = [
        ('risk_aversion', (-1, 1), 0.0, r"$\omega$"),
        ('distortion', (0, 2), 1.0, r"$\alpha$"),
        ('precision', (0, 3.5), False, r"$\lambda$")
    ]

    if class_model.__name__ == "AgentSideAdditive":
        args.append(
            ('side_bias', (-3, +10), 0.0, r"$\gamma$")
        )

    for i, arg in enumerate(args):

        pr, y_lim, mid_line, param_name = arg

        if np.min(fit[pr]) < y_lim[0] or np.max(fit[pr]) > y_lim[1]:

            msg = f"Some values are outside of range " \
                  f"for the 'history_best_param' plot " \
                  f"for parameter {pr} " \
                  f"(min: {np.min(fit[pr]):.2f}, max: {np.max(fit[pr]):.2f})"
            warnings.warn(msg)

        _plot_history_best_param(
            ax=axes[i],
            param=fit[pr],
            y_lim=y_lim,
            mid_line=mid_line,
            param_name=param_name)

        if 'regression' in data.keys():
            regression_param = data['regression']

            alpha, beta, relevant = regression_param[pr]

            n = len(fit[pr])
            x = np.arange(1, n+1)
            y = alpha + beta * x

            # print(alpha, beta, relevant)
            if relevant:
                line_style = "-"
                alpha = 1
            else:
                line_style = ":"
                alpha = 0.4
            axes[i].plot(x, y, linestyle=line_style,
                         alpha=alpha)
