from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time

start_time = time.time()
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa']
datasets =['2017-11-01-aa']

FIFIELD = True
INTERVAL_MAS = False
Bootstrapping = False
INTERVAL_REC = False
SOUND = False
VANROSSUM = False
GAP = False
ISI = False

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
    #mf.fifield_spike_detection(datasets[0])
    th = 2
    spike_count, fi_field, fsl = mf.fifield_analysis2(datasets[0], th, plot_fi=True)
    embed()

if Bootstrapping:
    # mf.resampling(datasets)
    nresamples = 10000
    mf.bootstrapping_vs(datasets, nresamples, plot_histogram=True)

if SOUND:
    spikes = mf.spike_times_indexes(datasets[0], 'Calls', th_factor=4, min_dist=50, maxph=0.8, show=False, save_data=True)


if VANROSSUM:
    dt_factor = 100
    # taus = [0.5, 1, 2, 5, 10, 20]  # in ms
    taus = [5]
    # duration = [40, 60, 80, 100, 120, 140]
    duration = [50, 100, 250, 500, 750, 1000, 1500, 2000]
    nsamples = 10
    for tt in taus:
        for dur in duration:
            mm = mf.vanrossum_matrix(datasets[0], tt/1000, dur/1000, dt_factor, boot_sample=nsamples,
                                     stim_type='series', save_fig=True)
    print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
    # mf.tagtostimulus(datasets[0])

if ISI:
    duration = 1000
    nsamples = 2
    mm = mf.isi_matrix(datasets[0], duration/1000, boot_sample=nsamples, stim_type='series', save_fig=True)

if GAP:
    tag_list = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Gap_tag_list.npy')
    sp = mf.spike_times_indexes(datasets[0], 'Gap', th_factor=2, min_dist=50, maxph=0.75, show=False)
    embed()

print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
