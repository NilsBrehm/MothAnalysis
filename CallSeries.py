from IPython import embed
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

# Font:
# matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 10

# Ticks:
matplotlib.rcParams['xtick.major.pad'] = '2'
matplotlib.rcParams['ytick.major.pad'] = '2'
matplotlib.rcParams['ytick.major.size'] = 4
matplotlib.rcParams['xtick.major.size'] = 4

# Title Size:
matplotlib.rcParams['axes.titlesize'] = 10

# Axes Label Size:
matplotlib.rcParams['axes.labelsize'] = 10

# Axes Line Width:
matplotlib.rcParams['axes.linewidth'] = 1

# Tick Label Size:
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

# Line Width:
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.color'] = 'k'

# Marker Size:
matplotlib.rcParams['lines.markersize'] = 2

# Error Bars:
matplotlib.rcParams['errorbar.capsize'] = 0

# Legend Font Size:
matplotlib.rcParams['legend.fontsize'] = 6

# Load data
samples = sio.loadmat('/media/nils/Data/Moth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]

plt.plot(samples[1])
plt.show()
embed()
