from parameters.parameters import SIG_STEEP, SIG_MID
# LABELS_CONTROL, \

from plot.tools.tools import scatter_and_sigmoid, add_text


def plot(ax, data, control_condition, color, dot_size=50, show_ylabel=True):
    """
    Produce the control sigmoid figures
    """

    x_label = "$EV_{right} - EV_{left}$"
    y_label = "p(choose right)"

    cd = control_condition

    d = data[cd]

    scatter_and_sigmoid(
        x=d['x'],
        y=d['y'],
        x_fit=d['fit']['x'],
        y_fit=d['fit']['y'],
        ax=ax,
        color=color,
        dot_size=dot_size
    )

    # title = LABELS_CONTROL[cd]
    # ax.set_title(title, fontsize=axis_label_font_size*1.2)

    ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)
    ax.axvline(0.0, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks((0, 0.5, 1))

    # Axis labels
    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel(y_label)

    # Remove top and right borders
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')

    txt = "$k=" + f"{data[cd]['fit'][SIG_STEEP]:.2f}" + "$" + "\n" \
        + "$x_0=" + f"{data[cd]['fit'][SIG_MID]:.2f}" + "$"

    # txt = \
    #     r"$F(x) = \dfrac{1}{1 + \exp(-k (x - x_0))}$" + "\n\n" \
    #     + "$k=" + f"{data[cd]['fit'][SIG_STEEP]:.2f}" + "$" + "\n" \
    #     + "$x_0=" + f"{data[cd]['fit'][SIG_MID]:.2f}" + "$" + "\n"
    add_text(ax, txt)
