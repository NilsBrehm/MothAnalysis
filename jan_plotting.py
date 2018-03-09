import matplotlib
import matplotlib.pyplot as plt
import seaborn
import matplotlib.colors as cols
import matplotlib.ticker as mtick

seaborn.set_context('paper')
seaborn.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 3})

matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['xtick.major.pad'] = '1'
matplotlib.rcParams['ytick.major.pad'] = '2'
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8

legend_font_size = 6
legend_marker_scale = 0.6
legend_handle_text_pad = -0.2
marker_size = 12
amp_color = "crimson"
punit_color = "dodgerblue"


def create_driven_response_plot(datasets, outfile):
    fig = plt.figure()
    fig_size = (20, 2)
    punit_stim_ax = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=2)
    punit_resp_ax = plt.subplot2grid(fig_size, (2, 0), rowspan=3, colspan=2)
    amp_stim_ax = plt.subplot2grid(fig_size, (7, 0), rowspan=2, colspan=2)
    amp_resp_ax = plt.subplot2grid(fig_size, (9, 0), rowspan=3, colspan=2)
    mean_var_ax = plt.subplot2grid(fig_size, (14, 0), rowspan=6, colspan=1)
    var_ax = plt.subplot2grid(fig_size, (14, 1), rowspan=6, colspan=1)

    amp_stim_ax.text(-0.18, 1.1, "B", transform=amp_stim_ax.transAxes, size=10)
    punit_stim_ax.text(-0.18, 1.1, "A", transform=punit_stim_ax.transAxes, size=10)
    var_ax.text(-0.5, 1.075, "D", transform=var_ax.transAxes, size=10)
    mean_var_ax.text(-0.5, 1.075, "C", transform=mean_var_ax.transAxes, size=10)

    amp_stim_ax.text(0.005, 1.05, "Ampullary stimulus", transform=amp_stim_ax.transAxes, size=8)
    amp_resp_ax.text(0.005, 1.05, "Responses", transform=amp_resp_ax.transAxes, size=8)
    punit_stim_ax.text(0.005, 1.05, "P-unit stimulus", transform=punit_stim_ax.transAxes, size=8)
    punit_resp_ax.text(0.005, 1.05, "Responses", transform=punit_resp_ax.transAxes, size=8)

    plot_driven_responses(datasets, punit_stim_ax, amp_stim_ax, punit_resp_ax, amp_resp_ax)
    punit_resp_ax.yaxis.set_label_coords(-0.1, 0.5)
    amp_resp_ax.yaxis.set_label_coords(-0.1, 0.5)

    plot_driven_response_variability(var_ax, mean_var_ax)
    var_ax.xaxis.set_label_coords(0.5, -0.2)
    mean_var_ax.tick_params(direction='out', pad=0)
    mean_var_ax.set_ylabel("resp. variability [Hz]")
    mean_var_ax.set_xlabel('')
    mean_var_ax.set_ylim([0, 150])
    mean_var_ax.yaxis.set_label_coords(-0.25, 0.5)

    seaborn.despine(fig=fig)

    fig.set_size_inches(3.2, 3.9)
    fig.subplots_adjust(left=0.15, top=0.965, bottom=0.085, right=0.97, wspace=0.75, hspace=1.75)

    if outfile:
        fig.savefig(outfile)
        plt.close()
    else:
        plt.show()