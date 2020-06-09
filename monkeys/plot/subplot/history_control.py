import numpy as np

from parameters.parameters import CONTROL_CONDITIONS, LABELS_CONTROL


def _plot_history_control(results, ax, color='C0', last=False,
                          ylabel="Success rate", title="Title", fontsize=10):

    # exemplary_d is a list (n=number of boxplot) of list (n=number of datapoints)
    n = len(results)

    tick_labels = [f"{i + 1}" for i in range(n)]

    # colors = ["black", color_gain, color_loss, color_gain, color_loss]
    positions = list(range(n))

    x_scatter = []
    y_scatter = []

    values_box_plot = []

    for i, res in enumerate(results):
        values_box_plot.append([])

        for v in results[i]:

            # For box plot
            values_box_plot[-1].append(v)

            # For scatter
            y_scatter.append(v)
            x_scatter.append(i + np.random.uniform(-0.025*n, 0.025*n))

    assert len(x_scatter) == len(y_scatter)
    ax.scatter(x_scatter, y_scatter, c=color, s=50, alpha=0.5,
               linewidth=0.0, zorder=1)

    ax.axhline(0.5, linestyle='--', color='0.3', zorder=-10, linewidth=0.5)
    ax.set_yticks(np.arange(0.0, 1.1, 0.5))
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(-0.02, 1.02)

    # Boxplot
    bp = ax.boxplot(values_box_plot, positions=positions,
                    labels=tick_labels, showfliers=False, zorder=2)

    # Warning: only one box, but several whiskers by plot
    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')
            # b.set_alpha(1)

    if not last:
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

    else:
        ax.set_xlabel('Chunk', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+1)


def plot(axes, hist_control_d):

    n_cond = len(CONTROL_CONDITIONS)

    for i, cond in enumerate(CONTROL_CONDITIONS):

        title = LABELS_CONTROL[cond]

        last = i == n_cond-1

        _plot_history_control(
            results=hist_control_d[cond],
            ax=axes[i],
            last=last,
            title=title,
        )