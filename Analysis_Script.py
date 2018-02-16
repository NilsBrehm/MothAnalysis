from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
datasets = ['2018-02-09-aa']

FilterSignalShow = False
FIFIELD = False
INTERVAL_MAS = False
Bootstrapping = False
INTERVAL_REC = False
SOUND = False
TEST = False
TEST2 = False
VANROSSUM = True

# Parameters for Spike Detection
peak_params = {'mph': 'dynamic', 'mpd': 40, 'valley': False, 'show': True, 'maxph': None, 'filter_on': True}


# Rect Intervals
if INTERVAL_REC:
    mf.rect_intervals_spike_detection(datasets, peak_params, True)  # Last param = show spike plot?
    mf.rect_intervals_cut_trials(datasets)

# Analyse Intervals MothASongs data stored on HDD
if INTERVAL_MAS:
    mf.moth_intervals_spike_detection(datasets, peak_params, False)  # Last param = show spike plot?
    mf.moth_intervals_analysis(datasets)

# Analyse FIField data stored on HDD
if FIFIELD:
    spike_threshold = 4
    mf.fifield_analysis(datasets, spike_threshold, peak_params)


if Bootstrapping:
    # mf.resampling(datasets)
    nresamples = 10000
    mf.bootstrapping_vs(datasets, nresamples, plot_histogram=True)

if FilterSignalShow:
    fs = 100*1000
    nyqst = 0.5*fs
    lowcut = 500
    highcut = 2000
    low = lowcut/nyqst
    high = highcut/nyqst

    y = mf.voltage_filter(datasets[0], [low, high], ftype='band', order=5, filter_on=True)
    y2 = mf.voltage_filter(datasets[0], [low, high], ftype='band', order=5, filter_on=False)
    plt.subplot(2, 1, 1)
    plt.plot(y2, 'k')
    plt.subplot(2, 1, 2)
    plt.plot(y, 'k')
    plt.show()

if SOUND:
    # mf.quickspikes_detection(datasets)
    # mf.soundfilestimuli_spike_detection(datasets, peak_params)
    # mf.soundfilestimuli_spike_distance(datasets)
    # mf.spike_distance_matrix(datasets)
    mf.get_spike_times(datasets[0], 'Calls', peak_params, show_detection=False)

if TEST:
    voltage = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-09-aa/DataFiles/Calls_voltage.npy').item()
    for k in range(80, len(voltage)):
        volt = voltage['SingleStimulus-file-'+str(k+1)][0]
        sp = mf.detect_peaks(volt, peak_params)

if TEST2:
    spikes = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-09-aa/DataFiles/Calls_spikes.npy').item()
    sp_1 = spikes['SingleStimulus-file-1'][0]
    sp_2 = spikes['SingleStimulus-file-10'][0]
    tau = 4
    dt_factor = 100
    d = mf.spike_train_distance(sp_1, sp_2, dt_factor, tau/1000, plot=True)
    print('Spike Train Distance = ' + str(d))

if VANROSSUM:
    dt_factor = 100
    # tau = 3
    for tau in np.arange(1, 21, 1):
        mf.vanrossum_matrix(datasets[0], tau/1000, dt_factor, template_choice=0)

    # mf.tagtostimulus(datasets[0])

print('Analysis done!')
