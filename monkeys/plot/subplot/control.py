import numpy as np

from parameters.parameters import CONTROL_CONDITIONS, \
    LABELS_CONTROL


def plot(ax, control_d, fontsize=10):
    """
    Produce the control figure
    """

    n = len(CONTROL_CONDITIONS)

    tick_labels = [LABELS_CONTROL[cd] for cd in CONTROL_CONDITIONS]
    colors = ['C0' for _ in range(n)]

    positions = list(range(n))

    x_scatter = []
    y_scatter = []
    colors_scatter = []

    values_box_plot = []

    for i, cond in enumerate(CONTROL_CONDITIONS):

        values_box_plot.append([])

        for v in control_d[cond]:
            # For box plot
            values_box_plot[-1].append(v)

            # For scatter
            y_scatter.append(v)
            x_scatter.append(i + np.random.uniform(-0.025*n, 0.025*n))
            colors_scatter.append(colors[i])

    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=50, alpha=0.5,
               linewidth=0.0, zorder=1)

    ax.axhline(0.5, linestyle='--', color='0.3', zorder=-10, linewidth=0.5)

    ax.set_yticks(np.arange(0.4, 1.1, 0.2))
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel("Success rate", fontsize=fontsize)
    ax.set_ylim(0.35, 1.02)

    # Boxplot
    bp = ax.boxplot(values_box_plot, positions=positions, labels=tick_labels,
                    showfliers=False, zorder=2)

    # Warning: only one box, but several whiskers by plot
    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')
            # b.set_alpha(1)

    ax.set_aspect(3)
