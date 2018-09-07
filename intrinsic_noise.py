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
from scipy.optimize import curve_fit


def extend_spike_train(sp, gap, extension_time, stim_type, path_names):
    if stim_type is 'series':
        stims = ['A7838.wav',
             'BCI1348.wav',
             'Chrostosoma_thoracicum.wav',
             'Creatonotos.wav',
             'Eucereon_appunctata.wav',
             'Eucereon_hampsoni.wav',
             'Eucereon_maia.wav',
             'GL005.wav',
             'Hyaleucera_erythrotelus.wav',
             'Hypocladia_militaris.wav',
             'PP241.wav',
             'PP612.wav',
             'PP643.wav',
             'Saurita.wav',
             'Uranophora_leucotelus.wav',
             'carales_PK1275.wav',
             'melese_PK1300_01.wav']
    else:
        print('Error. Wrong stim type')
        return 0
    c1, c2 = mf.tagtostimulus(path_names)
    tags = [[]] * len(stims)
    spikes = {}
    for j in range(len(stims)):
        tags[j] = c1['callseries/moths/' + stims[j]]
        spikes.update({tags[j]: sp[tags[j]]})
    extended_spikes = {}
    for key in spikes:
        a = spikes[key]
        dummy = [[]] * len(a)
        for i in range(len(a)):
            b = a[i]
            if len(b) == 0:
                print(c2[key] + ': Found empty spike train')
                b = np.nan
            else:
                while np.max(b) <= extension_time:
                    b = np.append(b, b + np.max(b) + gap)
            dummy[i] = b
        extended_spikes.update({key: dummy})
    return extended_spikes


def fit_function(x_fit, data, x_plot):
    def func(xx, a, c):
        # return a * np.exp(-d * x1) + c
        # return bottom + ((top-bottom)/(1+np.exp((V50-xx)/Slope)))
        return a*xx+c

    p0 = [100, 1000]
    # bounds = (0, [np.max(data)/2+10, np.max(data)*2+10, np.max(x_fit), np.inf])
    # print('p0:')
    # print(p0)
    # print('bounds:')
    # print(bounds)
    # print('-----------------')
    # popt, pcov = curve_fit(func, x_fit, data, p0=p0, maxfev=10000,  bounds=bounds)
    popt, pcov = curve_fit(func, x_fit, data, p0=p0)

    # x = np.linspace(np.min(x_fit), np.max(x_fit), 1000)
    x = np.linspace(x_plot[0], x_plot[1], 1000)
    y = func(x, *popt)
    y0 = func(popt[-2], *popt)
    perr = np.sqrt(np.diag(pcov))
    return x, y, popt, perr, y0


path_names = mf.get_directories('2018-02-16-aa')

spikes = np.load(path_names[1] + 'Calls_spikes.npy').item()
# np.random.normal(0, np.std(a)*0.05, 10)

c1, c2 = mf.tagtostimulus(path_names)
call_name = 'callseries/moths/' + 'carales_PK1275.wav'
stim = c1[call_name]

spike_times = spikes[stim]

extended_trains = mf.extend_spike_train(spikes, gap=0.05, extension_time=4, stim_type='series', path_names=path_names)
embed()
exit()
# Add noise
noise = [[]] * len(spike_times)
# trains = [[]] * len(spike_times)
epulses = [[]] * len(spike_times)
tau = 0.001
dt = 0.001
percent_noise = np.arange(0, 0.01, 0.001)
vr = [[]] * len(percent_noise)
std_trains = np.std(np.concatenate(spike_times))
std_noise = std_trains * percent_noise

for nn in range(len(percent_noise)):
    for i in range(len(spike_times)):
        noise[i] = abs(spike_times[i] + np.random.normal(0, std_noise[nn], len(spike_times[i])))
        # trains[i] = spk.SpikeTrain(noise[i], [0, 1])
        epulses[i] = mf.fast_e_pulses(noise[i], tau, dt)

    d_vr = np.zeros(shape=(len(epulses), len(epulses)))
    for k in range(len(epulses)):
        for j in range(len(epulses)):
            d_vr[k, j] = mf.vanrossum_distance(epulses[k], epulses[j], dt, tau)
    vr[nn] = np.mean(d_vr)**2


x, y, popt, perr, y0 = fit_function(std_noise**2, vr, [-0.0001, 0.0001])
plt.plot(std_noise**2, vr, 'ko')
plt.plot(x, y, 'k')
plt.show()
embed()
exit()
