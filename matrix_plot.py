import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
from IPython import embed
import seaborn
from mpl_toolkits import axes_grid1

seaborn.set_context('paper')
seaborn.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 3})

matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['xtick.major.pad'] = '1'
matplotlib.rcParams['ytick.major.pad'] = '2'
matplotlib.rcParams['axes.titlesize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 6
matplotlib.rcParams['ytick.labelsize'] = 6
matplotlib.rcParams['axes.labelsize'] = 6

legend_font_size = 6
legend_marker_scale = 0.6
legend_handle_text_pad = -0.2
marker_size = 12

# Load Data
data = sio.loadmat('/media/brehm/Data/AvsP.mat')
m = data['MaxCorr_AP']

fig = plt.figure()
fig_size = (2, 5)
ax1 = plt.subplot2grid(fig_size, (0, 0), rowspan=2)
ax2 = plt.subplot2grid(fig_size, (0, 1), rowspan=2)
ax3 = plt.subplot2grid(fig_size, (0, 2), colspan=3)
ax4 = plt.subplot2grid(fig_size, (1, 2))
ax5 = plt.subplot2grid(fig_size, (1, 3))
ax6 = plt.subplot2grid(fig_size, (1, 4))

ax4.pcolor(m)
ax4.set(title='AvsP', ylabel='active pulses', xlabel='passive pulses')
ax4.set_aspect('equal')


ax5.pcolor(m)
ax5.set(title='AvsA', ylabel='active pulses', xlabel='active pulses')
ax5.set_aspect('equal')

cf = ax6.pcolor(m)
ax6.set(title='PvsP', ylabel='passive pulses', xlabel='passive pulses')
ax6.set_aspect('equal')
fig.colorbar(cf, ax=ax6)

plt.tight_layout()
seaborn.despine(fig=fig)

fig.set_size_inches(6, 4)  # in inches: breite x hoehe
# fig.subplots_adjust(left=0.15, top=0.965, bottom=0.085, right=0.97, wspace=0.75, hspace=1.75)

plt.show()
