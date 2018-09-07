import pyspike as spk
import scipy.io as sio
from tqdm import tqdm
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from joblib import Parallel,delayed
import myfunctions as mf


def isListEmpty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    return False  # Not a list


def poission_spikes(trials, rate, tmax):
    """Homogeneous Poission Spike Generator

       Notes
       ----------
       Generate spike times of a homogeneous poisson process using the exponential interspike interval distribution.

       Parameters
       ----------
       trials: Number of trials
       rate: Firing Rate in Hertz
       tmax: Max duration of spike trains in seconds

       Returns
       -------
       Spike Times (in seconds)

       """
    spikes = [[]] * trials
    # spike_trains = [[]] * trials
    intervals = [[]] * trials
    mu = 1/rate
    nintervals = 2 * np.round(tmax/mu)
    for k in range(trials):
        # Exponential random numbers
        intervals[k] = np.random.exponential(mu, size=int(nintervals))
        times = np.cumsum(intervals[k])
        spikes[k] = list(times[times <= tmax])
        if len(spikes[k]) == 0:
            spikes[k] = [0]
        # spike_trains[k] = spk.SpikeTrain(spikes[k], [0, tmax])
    return spikes, intervals


trials = 10
rate = 100
tmax = 0.1
durations = np.arange(0, 1000, 10)
durations[0] = 1
durations = durations / 1000
spikes = [[]] * len(durations)
spike_trains = [[]] * len(durations)
intervals = [[]] * len(durations)
d_isi = [[]] * len(durations)

# for i in range(len(durations)):
#     spikes[i], intervals[i] = poission_spikes(trials, rate, durations[i])
#     spike_trains[i] = [[]] * len(spikes[i])
#     for k in range(len(spikes[i])):
#         spike_trains[i][k] = spk.SpikeTrain(spikes[i][k], [0, durations[i]])
#     d_isi[i] = spk.isi_distance(spike_trains[i])

base_tmax = [0.05, 0.1, 0.5, 1, 2]
t_diffs = np.arange(0, 2, 0.001)
isi = [[]] * len(base_tmax)
vr = [[]] * len(base_tmax)
dt = 0.001
tau = 0.01

for k in range(len(base_tmax)):
    # d_isi, d_vr = [[[]] * len(t_diffs)] * 2
    d_isi = [[]] * len(t_diffs)
    d_vr = [[]] * len(t_diffs)
    for i in range(len(t_diffs)):
        p01, _ = poission_spikes(1, rate, tmax=base_tmax[k])
        p02, _ = poission_spikes(1, rate, tmax=base_tmax[k]+t_diffs[i])
        spike_train01 = spk.SpikeTrain(p01[0], [0, base_tmax[k]+t_diffs[i]])
        spike_train02 = spk.SpikeTrain(p02[0], [0, base_tmax[k]+t_diffs[i]])
        pulse01 = mf.fast_e_pulses(np.array(p01[0]), tau=tau, dt=dt)
        pulse02 = mf.fast_e_pulses(np.array(p02[0]), tau=tau, dt=dt)

        d_isi[i] = spk.isi_distance(spike_train01, spike_train02)
        d_vr[i] = mf.vanrossum_distance(pulse01, pulse02, dt=dt, tau=tau)
    isi[k] = d_isi
    vr[k] = d_vr

for j in range(len(base_tmax)):
    plt.subplot(121)
    plt.plot(t_diffs, vr[j], 'o', label='base: ' + str(base_tmax[j]))
    plt.subplot(122)
    plt.plot(t_diffs, isi[j], 'o', label='base: ' + str(base_tmax[j]))

plt.legend()
plt.show()

# Estimate for two poisson trains with same rate and duration
b = [(rate * base_tmax[i]) for i in range(len(vr))]

embed()
exit()