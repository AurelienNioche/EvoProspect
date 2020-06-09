from parameters.parameters import \
    CONTROL_CONDITIONS, LABELS_CONTROL, \
    SIG_STEEP, \
    SIG_MID

from plot.tools.tools import scatter_and_sigmoid, add_text


def plot(axes, data,
         axis_label_font_size=20,
         ticks_label_font_size=14):
    """
    Produce the control sigmoid figures
    """

    x_label = "$EV_{right} - EV_{left}$"
    y_label = "F(Choose right)"

    for i, cd in enumerate(CONTROL_CONDITIONS):
        ax = axes[i]

        d = data[cd]

        scatter_and_sigmoid(
            x=d['x'],
            y=d['y'],
            x_fit=d['fit']['x'],
            y_fit=d['fit']['y'],
            ax=ax)

        title = LABELS_CONTROL[cd]
        ax.set_title(title, fontsize=axis_label_font_size*1.2)

        ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)
        ax.axvline(0.0, alpha=0.5, linewidth=1, color='black',
                   linestyle='--', zorder=-10)

        ax.set_ylim(-0.01, 1.01)
        ax.set_yticks((0, 0.5, 1))

        # Axis labels
        ax.set_xlabel(x_label, fontsize=axis_label_font_size)
        ax.set_ylabel(y_label, fontsize=axis_label_font_size)

        # Remove top and right borders
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_color('none')

        ax.tick_params(axis='both', which='major',
                       labelsize=ticks_label_font_size)
        ax.tick_params(axis='both', which='minor',
                       labelsize=ticks_label_font_size)

        txt = \
            r"$F(x) = \dfrac{1}{1 + \exp(-k (x - x_0))}$" + "\n\n" \
            + "$k=" + f"{data[cd]['fit'][SIG_STEEP]:.2f}" + "$" + "\n" \
            + "$x_0=" + f"{data[cd]['fit'][SIG_MID]:.2f}" + "$" + "\n"
        add_text(ax, txt)
