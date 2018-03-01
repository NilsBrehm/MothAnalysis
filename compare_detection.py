import myfunctions as mf
import numpy as np
import quickspikes as qs
import matplotlib.pyplot as plt
from IPython import embed

# Load Data
voltage = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-09-aa/DataFiles/Calls_voltage.npy').item()
data = voltage['SingleStimulus-file-2'][0]

# Filter
fs = 100 * 1000
nyqst = 0.5 * fs
lowcut = 300
highcut = 2000
low = lowcut / nyqst
high = highcut / nyqst
x = mf.voltage_trace_filter(data, [low, high], ftype='band', order=4, filter_on=True)

# Spike Detection Methods
sp1 = mf.indexes(x, th_factor=2, min_dist=50, maxph=0.8)

peak_params = {'mph': 'dynamic', 'mpd': 50, 'valley': False, 'show': False, 'maxph': None, 'filter_on': False}
sp2 = mf.detect_peaks(x, peak_params)

reldet = qs.detector(2, 500)
reldet.scale_thresh(x.mean(), x.std())
# reldet.scale_thresh(np.median(x), np.median(abs(x)/0.6745))
sp3 = reldet.send(x)

# Plot
plt.subplot(3, 1, 1)
plt.plot(x, 'k')
plt.plot(sp1, x[sp1], 'ro')
plt.title('Indexes()')

plt.subplot(3, 1, 2)
plt.plot(x, 'k')
plt.plot(sp2, x[sp2], 'ro')
plt.title('PeakSeek()')

plt.subplot(3, 1, 3)
plt.plot(x, 'k')
plt.plot(sp3, x[sp3], 'ro')
plt.title('QuickSpikes()')

plt.show()
