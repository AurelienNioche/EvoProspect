from parameters.parameters import SIG_STEEP, SIG_MID
from plot.tools.tools import scatter_and_sigmoid, add_text


def plot(ax, data, color='C0',
         axis_label_font_size=20,
         ticks_label_font_size=14):
    """
    Produce the figure which presents at which extent
    the risky option was chosen according to the difference
    between expected values,
    i.e. the certainty-risk trade-off figure
    """

    scatter_and_sigmoid(
        x=data['x'], y=data['y'],
        x_fit=data['fit']['x'], y_fit=data['fit']['y'],
        color=color,
        ax=ax)

    ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)
    ax.axvline(0, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)

    # ax.set_ylim(-0.01, 1.01)
    ax.set_yticks((0, 0.5, 1))

    # Axis labels
    ax.set_xlabel(
        r"$EV_{\mathrm{Riskiest\,option}} - EV_{\mathrm{Safest\,option}}$",
        fontsize=axis_label_font_size)
    ax.set_ylabel(
        "F(Choose riskiest option)",
        fontsize=axis_label_font_size)

    # Remove top and right borders
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_color('none')

    ax.tick_params(axis='both', which='major', labelsize=ticks_label_font_size)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_label_font_size)

    txt = \
        r"$F(x) = \dfrac{1}{1 + \exp(-k (x - x_0))}$" + "\n\n" \
        + "$k=" + f"{data['fit'][SIG_STEEP]:.2f}" + "$" + "\n"\
        + "$x_0=" + f"{data['fit'][SIG_MID]:.2f}" + "$" + "\n" \

    add_text(ax, txt)
