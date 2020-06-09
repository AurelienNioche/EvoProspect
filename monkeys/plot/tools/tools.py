import numpy as np
from string import ascii_lowercase  #ascii_uppercase


def scatter_and_sigmoid(ax, x, y, x_fit, y_fit, color='C0', label=None,
                        line_width=3, point_size=100, alpha_scatter=0.5):

    if label is not None:
        label = label.capitalize()

    if x_fit is not None and y_fit is not None:

        ax.plot(x_fit, y_fit, color=color, linewidth=line_width, label=label)

    ax.scatter(x, y, color=color, alpha=alpha_scatter, s=point_size)


def add_text(ax, txt,):
    ax.text(0.05, 0.9, txt,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)


def add_letter(ax, i):
    ax.text(-0.1, 1.1, ascii_lowercase[i],
            transform=ax.transAxes, size=20, weight='bold')


def scatter_boxplot(data, ax, y_label, x_tick_label, title,
                    color='C0', err=None, dot_size=20):

    # For scatter
    x_scatter = np.random.uniform(-0.05, 0.05, size=len(data))

    # Plot the scatter
    if err is None:
        ax.scatter(x_scatter, data, c=color, s=dot_size,
                   alpha=0.2, linewidth=0.0, zorder=1)
    else:
        ax.errorbar(x_scatter, data, yerr=err, fmt='o', alpha=0.2, zorder=1,
                    elinewidth=0.5)

    # Plot the boxplot
    bp = ax.boxplot(data, positions=[0, ],
                    labels=[x_tick_label, ], showfliers=False, zorder=2)

    # Set the color of the boxplot
    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')

    # Set the label of the y axis
    ax.set_ylabel(y_label)

    # Set the title
    ax.set_title(title)