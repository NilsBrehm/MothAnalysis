import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import myfunctions as mf

"""
# Load data
datasets = ['2017-12-05-ab']
data_name = datasets[0]
pathname = "figs/" + data_name + "/"
fname = pathname + 'intervals_mas_spike_times.npy'
voltage = np.load(fname).item()

dt = 100 * 100
# spike_times1 = np.array([2, 3, 5, 7, 10, 12, 14, 18])
# spike_times2 = np.array([2, 3, 5, 7, 8, 9, 10, 12, 14, 18])
spike_times1 = voltage[0]['all_spike_times']
spike_times2 = voltage[1]['all_spike_times']

tau = 1
duration1 = int(spike_times1[-1] * dt)
duration2 = int(spike_times2[-1] * dt)

mf.spike_train_distance(spike_times1, spike_times2, dt, duration1, duration2, tau)
"""
"""
dt = 100 * 1000
t = np.arange(0, 1, 1/dt)
tau = 0.1  # tau in seconds!

spike_times1 = [0.1, 0.5, 0.6, 0.8]
spike_times2 = [0.1, 0.5, 0.52, 0.55, 0.6, 0.8]
f = np.zeros(len(t))
g = np.zeros(len(t))

for ti in spike_times1:
    dummy = np.heaviside(t-ti, 0) * np.exp(-(t-ti)/tau)
    f = f + dummy

for ti in spike_times2:
    dummy = np.heaviside(t-ti, 0) * np.exp(-(t-ti)/tau)
    g = g + dummy

plt.subplot(2, 1, 1)
plt.plot(t, f)
plt.ylabel('f(t)')

plt.subplot(2, 1, 2)
plt.plot(t, g)
plt.xlabel('Time [s]')
plt.ylabel('g(t)')

plt.show()

D = np.sum((f - g) ** 2) / (tau * dt)  # For this formula tau must be in samples!
print('Spike Train Difference: %s' % D)
print('Tau = %s' % (tau / dt))
"""
# Load data
datasets = ['2017-12-05-ab']
data_name = datasets[0]
pathname = "figs/" + data_name + "/"
fname = pathname + 'intervals_mas_spike_times.npy'
voltage = np.load(fname).item()

spike_times1 = voltage[0]['all_spike_times']
# spike_times2 = spike_times1.copy()
# spike_times2[-2] = spike_times2[-2] + 0.05
spike_times2 = voltage[1]['all_spike_times']
dt = 100 * 1000
tau = 0.005
duration = np.round(np.max([np.max(spike_times1), np.max(spike_times2)]), 3)
D = mf.spike_train_distance(spike_times1, spike_times2, dt, duration, tau)

embed()
