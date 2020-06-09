def fig_info(ax, info):

    ax.text(0.5, 0.5, info.text, fontsize=15, wrap=True, ha='center',
            va='center')

    ax.set_axis_off()
