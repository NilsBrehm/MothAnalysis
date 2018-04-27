import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

np.random.seed(sum(map(ord, "aesthetics")))


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

settings = \
    {'axes.axisbelow': True,
     'axes.edgecolor': '.8',
     'axes.facecolor': 'white',
     'axes.grid': True,
     'axes.labelcolor': '.15',
     'axes.linewidth': 1.0,
     'figure.facecolor': 'white',
     'font.family': [u'sans-serif'],
     'font.sans-serif': [u'Arial',
                         u'DejaVu Sans',
                         u'Liberation Sans',
                         u'Bitstream Vera Sans',
                         u'sans-serif'],
     'grid.color': '.8',
     'grid.linestyle': u'-',
     'image.cmap': u'rocket',
     'legend.frameon': False,
     'legend.numpoints': 1,
     'legend.scatterpoints': 1,
     'lines.solid_capstyle': u'round',
     'text.color': '.15',
     'xtick.color': '.15',
     'xtick.direction': u'out',
     'xtick.major.size': 0.0,
     'xtick.minor.size': 0.0,
     'ytick.color': '.15',
     'ytick.direction': u'out',
     'ytick.major.size': 0.0,
     'ytick.minor.size': 0.0}

# sns.axes_style()
# sns.set_style({'axes.grid': True, 'grid.color': '.5',  'axes.facecolor': '.2'})
sns.set_style(settings)

sinplot()
sns.despine(offset=10, trim=True)
plt.show()

# with sns.axes_style("ticks"):
#     plt.subplot(211)
#     sinplot()
# plt.subplot(212)
# sinplot(-1)
# plt.show()

plt.style.use(['seaborn-white', 'seaborn-paper'])
# plt.rc("font", family="Arial")
sinplot()
sns.despine(offset=10, trim=True)
plt.show()
embed()


