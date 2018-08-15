import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython import embed
import myfunctions as mf


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

settings = \
    {'axes.axisbelow': True,
     'axes.edgecolor': 'black',
     'axes.facecolor': 'white',
     'axes.grid': False,
     'axes.labelcolor': '.15',
     'axes.linewidth': 2.0,
     'figure.facecolor': 'white',
     'font.family': [u'sans-serif'],
     'font.sans-serif': [u'Arial',
                         u'DejaVu Sans',
                         u'Liberation Sans',
                         u'Bitstream Vera Sans',
                         u'sans-serif'],
     'grid.color': 'black',
     'grid.linestyle': u'-',
     'image.cmap': u'rocket',
     'legend.frameon': False,
     'legend.numpoints': 1,
     'legend.scatterpoints': 1,
     'lines.solid_capstyle': u'round',
     'text.color': 'black',
     'xtick.color': 'black',
     'xtick.direction': u'out',
     'xtick.major.size': 5.0,
     'xtick.minor.size': 2.0,
     'ytick.color': 'black',
     'ytick.direction': u'out',
     'ytick.major.size': 5.0,
     'ytick.minor.size': 2.0}

# sns.axes_style()
# sns.set_style({'axes.grid': True, 'grid.color': '.5',  'axes.facecolor': '.2'})
# sns.set_style(settings)

# sns.set_context('paper')
# sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
#
# matplotlib.rcParams.update({'font.size': 8})
# matplotlib.rcParams['xtick.major.pad'] = '2'
# matplotlib.rcParams['ytick.major.pad'] = '2'
# matplotlib.rcParams['axes.titlesize'] = 10
# matplotlib.rcParams['axes.labelsize'] = 10
# matplotlib.rcParams['xtick.labelsize'] = 8
# matplotlib.rcParams['ytick.labelsize'] = 8

# Import Default Settings
mf.plot_settings()

fig = plt.figure()
plt.subplot(211)
sinplot()
plt.xlabel('X label')
plt.ylabel('Y label')

plt.subplot(212)
sinplot()
plt.xlabel('X label')
plt.ylabel('Y label')

#sns.despine(offset=10, trim=True)
sns.despine(fig=fig)
fig.set_size_inches(3.2, 3.9)
fig.subplots_adjust(left=0.2, top=0.98, bottom=0.2, right=0.98, wspace=0.4, hspace=0.4)
plt.show()

# with sns.axes_style("ticks"):
#     plt.subplot(211)
#     sinplot()
# plt.subplot(212)
# sinplot(-1)
# plt.show()

# plt.style.use(['seaborn-white', 'seaborn-paper'])
# # plt.rc("font", family="Arial")
# sinplot()
# sns.despine(offset=10, trim=True)
# plt.show()


