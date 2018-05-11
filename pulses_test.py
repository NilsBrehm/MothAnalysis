import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import time
# Testing Old vs New Method for computing e pulses
# New method seems to be a lot faster, especially for long spike trains


def spike_e_pulses(spikes, dt_factor, tau, t_max, method):
    # This is the old method
    # tau in seconds
    # dt_factor: dt = tau/dt_factor
    # spike times in seconds
    # duration in seconds
    pulses = [[]] * len(spikes)
    for k in range(len(spikes)):
        spike_times = spikes[k]
        dt = tau / dt_factor
        t = np.arange(0, t_max - dt, dt)
        f = np.zeros(len(t))
        for ti in spike_times:
            if method == 'exp':
                dummy = np.heaviside(t - ti, 0) * np.exp(-((t - ti)*np.heaviside(t - ti, 0)) / tau)
            elif method == 'rect':
                dummy = np.heaviside(t - ti, 0) * np.heaviside((ti+tau)-t, 0)
            else:
                print('Method not Found')
                exit()
            f = f + dummy
        pulses[k] = f
    return pulses


def new_e_pulses(spikes, tau, dt_factor, t_max):
    # This is the new method
    def exp_kernel(ta, step):
        x = np.arange(0, tau*5, step)
        y = np.exp(-x/ta)
        return y
    dt = tau / dt_factor
    pulses = [[]] * len(spikes)
    for k in range(len(spikes)):
        spike_times = spikes[k][spikes[k] < t_max]
        t = np.arange(0, t_max - dt, dt)
        r = np.zeros(len(t))
        spike_ids = np.round(spike_times / dt) - 2
        r[spike_ids.astype('int')] = 1
        kernel = exp_kernel(tau, dt)
        pulses[k] = np.convolve(r, kernel, mode='full')
    return pulses


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
    intervals = [[]] * trials
    mu = 1/rate
    nintervals = 2 * np.round(tmax/mu)
    for k in range(trials):
        # Exponential random numbers
        intervals[k] = np.random.exponential(mu, size=int(nintervals))
        times = np.cumsum(intervals[k])
        spikes[k] = times[times <= tmax]
    return spikes, intervals


def vanrossum_distance(f, g, dt_factor, tau):

    # Make sure both trains have same length
    difference = abs(len(f) - len(g))
    if len(f) > len(g):
        g = np.append(g, np.zeros(difference))
    elif len(f) < len(g):
        f = np.append(f, np.zeros(difference))

    # Compute VanRossum Distance
    dt = tau/dt_factor
    dist = np.sum((f - g) ** 2) * (dt / tau)
    # f_count = np.sum(f)* (dt / tau)
    # g_count = np.sum(g) * (dt / tau)
    return dist

trials = 100
t_max = 1
spikes, intervals = poission_spikes(trials, 100, t_max)

tau = 0.001
dt_factor = 100
a = [[]] * 2
b = [[]] * 2
for test in range(2):
    t1 = time.time()
    if test == 0:
        pulses = new_e_pulses(spikes, tau, dt_factor, t_max)
        info = 'New'
    else:
        pulses = spike_e_pulses(spikes, dt_factor, tau, t_max, method='exp')
        info = 'Old'
    a[test] = pulses[0]
    t2 = time.time()
    d = np.zeros(shape=(len(pulses), len(pulses)))
    for k in range(len(pulses)):
        for i in range(len(pulses)):
            d[k, i] = vanrossum_distance(pulses[k], pulses[i], dt_factor, tau)
    t3 = time.time()

    b[test] = t2-t1
    print(info)
    print('pulses ' + str(t2-t1))
    print('distance ' + str(t3-t3))
    print('')
    print('total ' + str(t3-t1))
    print('')
    print('distance = ' + str(d[0, 2]))
    print('---------------')

print('new is ' + str(np.round(b[1]/b[0])) + ' times faster')

ax1 = plt.subplot(2,1,1)
ax1.plot(a[0])
ax1.set_xlim(0, 5000)
ax1.set_xticks(np.arange(0, 5000, 500))

ax2 = plt.subplot(2,1,2)
ax2.plot(a[1])
ax2.set_xlim(0, 5000)
ax2.set_xticks(np.arange(0, 5000, 500))
plt.show()
embed()

