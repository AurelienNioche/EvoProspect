from parameters.parameters import SIG_STEEP, SIG_MID
from plot.tools.tools import scatter_and_sigmoid, add_text


def plot(ax, data, color='C0', show_ylabel=True, dot_size=50):
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
        ax=ax,
        dot_size=dot_size
    )

    ax.axhline(0.5, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)
    ax.axvline(0, alpha=0.5, linewidth=1, color='black',
               linestyle='--', zorder=-10)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks((0, 0.5, 1))

    # Axis labels
    ax.set_xlabel(
        r"$EV_{riskiest} - EV_{safest}$")

    if show_ylabel:
        ax.set_ylabel(
            "p(choose riskiest)")

    # Remove top and right borders
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')

    txt = "$k=" + f"{data['fit'][SIG_STEEP]:.2f}" + "$" + "\n"\
        + "$x_0=" + f"{data['fit'][SIG_MID]:.2f}" + "$" \

    # txt = \
    #     r"$F(x) = \dfrac{1}{1 + \exp(-k (x - x_0))}$" + "\n\n" \
    #     + "$k=" + f"{data['fit'][SIG_STEEP]:.2f}" + "$" + "\n"\
    #     + "$x_0=" + f"{data['fit'][SIG_MID]:.2f}" + "$" + "\n" \

    add_text(ax, txt)
