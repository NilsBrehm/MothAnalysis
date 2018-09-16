import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm
from joblib import Parallel,delayed
import os
import seaborn as sns
import pickle
import matplotlib
from matplotlib.colors import LogNorm
import scipy.io.wavfile as wav
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects
from scipy import signal
import scipy
import scipy.io as sio
import pyspike as spk

start_time = time.time()
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa']  # Calls Creatonotos
# datasets = ['2018-02-20-aa']  # Calls Estigmene
# datasets = ['2017-12-05-aa']  # FI
# datasets = ['2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa', '2017-11-01-aa', '2017-11-16-aa']  # Carales FIs
# dat = ['2017-11-25-aa', '2017-11-25-ab', '2017-11-27-aa', '2017-11-29-aa', '2017-12-01-aa', '2017-12-05-ab',
#        '2017-11-14-aa', '2017-11-16-aa', '2017-11-17-aa', '2017-12-01-ac', '2018-02-16-aa', '2018-02-20-aa']  # GAP

# dat = ['2017-11-27-aa', '2018-02-16-aa']  # good GAP
#
# datasets = [dat[0]]
# print(datasets)

TEST = False

CALLS = True
CALL_STRUC = False
CALL_STATS = False

Bootstrapping = False

# Compute Intervall stuff
INTERVAL_MAS = False
INTERVAL_REC = False
INTERVAL_REC_SPONT = False
OVERALLVS = False

GAP = False
SOUND = False
POISSON = False

# Compute Van Rossum Distance
EPULSES = False
VANROSSUM = False
MVSB = False
PLOT_MVSB = False
PLOT_MVSB_DPRIME = False

# Compute other Distances
ISI = False
DISTANCE_RATIOS = False

# Other Distances Correct Matches
PLOT_CORRECT = False
PLOT_CORRECT_OVERALL = False
PLOT_DISTANCES_CORRECT = False

# Ratio: within vs. between distances
PLOT_D_RATIOS = False
PLOT_D_RATIOS_OVERALL = False

# -------------
# PLOTs
# Plot Stimulus Calls
CALLSFROMMATLAB = False
CALLSERIESFROMMATLAB = False
PLOT_CALLS = False
CUMHIST = False

# VanRossum Tau vs Duration
PLOT_VR_SPIKEMATCHING = False
PLOT_VR_TAUVSDUR = False
PLOT_VR_TAUVSDUR_OVERALL = False

# VanRossum Matched Spike Trains with different taus and durations
PLOT_VR = False

# Moth vs Bat: d prime and percentage
PLOT_MvsB = False

# Other Distances Matched Spike Trains with different durations
PLOT_DISTANCES = False

# Pulse Train Stuff
PULSE_TRAIN_VANROSSUM = False
PULSE_TRAIN_ISI = False

# FI Stuff
FIFIELD = False
FI_OVERANIMALS = False

# Rate and SYNC Correlation with Stimnulus (Rect and Pulses)
PLOT_CORRS = False

# Select recordings from csv file
SELECT = True

# **********************************************************************************************************************
# Settings for Spike Detection =========================================================================================
th_factor = 3
mph_percent = 1
bin_size = 0.001
# If true show plots (list: 0: spike detection, 1: overview, 2: vector strength)
show = False

# Settings for Call Analysis ===========================================================================================
# General Settings
save_extended_spikes = False
extended = False
POISSON_TRAINS = False
POISSON_TAU_CORRECT = True

# stim_type = 'moth_single_selected'
# stim_type = 'all_single'
stim_type = 'poisson_2diff'
stim_length = 'series'
if stim_length is 'single':
    # duration = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    duration = list(np.arange(0, 255, 5))
    duration[0] = 1
if stim_length is 'series':
    # duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    duration = list(np.arange(0, 2550, 50))
    duration[0] = 10
# ISI and co
# profs = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR', 'VanRossum']
profs = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR']
profs_plot_correct = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR', 'VanRossum']

# Bootstrapping
nsamples = 10

# VanRossum
method = 'exp'
dt = 0.001
# taus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
taus = list(np.concatenate([np.arange(1, 21, 1), np.arange(30, 105, 5), np.arange(200, 1000, 55)]))
taus.append(1000)

# ======================================================================================================================

# Select data
if SELECT:
    import csv
    p = os.path.join('..', 'overview.csv')
    with open(p, newline='') as f:
        datasets = []
        reader = csv.reader(f)
        for row in reader:
            if INTERVAL_MAS:
                if row[3] == 'True':  # this is MAS
                    datasets.append(row[0])
            if INTERVAL_REC:
                if row[2] == 'True':  # this is RectIntervals
                    datasets.append(row[0])
            if GAP:
                if row[1] == 'True':  # this is GAP
                    datasets.append(row[0])
            if FIFIELD:
                if row[4] == 'True' and row[6] == 'Estigmene':  # this is FI
                    datasets.append(row[0])
            if CALLS:
                if row[5] == 'True':  # this is calls
                    datasets.append(row[0])
    datasets = sorted(datasets)

# Get relative paths ===================================================================================================
# RECT: good recs: all Estigmene
# datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2018-02-16-aa', '2017-12-05-ab']

# print('data set count: ' + str(len(datasets)))

# data_name = datasets[4]
# print(data_name)
#
# #  path_names = [data_name, data_files_path, figs_path, nix_path]
# path_names = mf.get_directories(data_name=data_name)
if PLOT_CALLS:
    # Plot calls
    mf.plot_settings()
    # p = '/media/brehm/Data/MasterMoth/stimuli_plotting/callseries/moths/'
    # p = '/media/brehm/Data/MasterMoth/stimuli_plotting/callseries/bats/'
    # p = '/media/brehm/Data/MasterMoth/stimuli_plotting/naturalmothcalls/'
    p = '/media/brehm/Data/MasterMoth/stimuli_plotting/batcalls/'

    listings = os.listdir(p)
    calls = [[]] * len(listings)
    for k in range(len(listings)):
            calls[k] = wav.read(p + listings[k])

    # Create Grid
    mf.plot_settings()
    fig = plt.figure()
    # fs = calls[0][0]
    # ids = [1, 3, 5, 6, 8]  # moth series
    # ids = [0, 1, 4, 5, 9]  # bat series
    # ids = [0, 1, 2, 3, 18]  # moth single
    ids = [0, 2, 3, 5, 10]  # bat single
    # ids = np.arange(0, len(calls), 1)

    grid = matplotlib.gridspec.GridSpec(nrows=len(ids), ncols=1)
    # fig = plt.figure(figsize=(5.9, 2.9))
    for i in range(len(ids)):
        fs = calls[ids[i]][0]
        t = np.arange(0, len(calls[ids[i]][1]) / fs, 1 / fs)
        ax = plt.subplot(grid[i])
        ax.plot(t[0:len(calls[ids[i]][1])], calls[ids[i]][1], 'k')
        ax.set_yticks([])
        ax.set_xlim(-0.01, 0.21)
        ax.set_xticks(np.arange(0, 0.21, 0.05))
        ax.set_xticklabels([])
    ax.set_xticklabels(np.arange(0, 0.21, 0.05))
    ax.set_xlabel('Time [s]')
    # fig.set_size_inches(5.9, 1.9)
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
    figname = "/media/brehm/Data/MasterMoth/figs/SingleCalls_Bats.pdf"
    fig.savefig(figname)
    plt.close(fig)

    exit()

if DISTANCE_RATIOS:
    # Try to load e pulses from HDD
    datasets = ['2018-02-20-aa', '2018-02-16-aa', '2018-02-09-aa']
    data_name = datasets[1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    # method = 'exp'
    p = path_names[1]
    if stim_length is 'series':
        if extended:
            spikes = np.load(p + 'Calls_spikes_extended.npy').item()
        else:
            spikes = np.load(p + 'Calls_spikes.npy').item()
    elif stim_length is 'single':
        if extended:
            spikes = np.load(p + 'Calls_spikes_extended.npy').item()
        else:
            spikes = np.load(p + 'Calls_spikes.npy').item()


    tag_list = np.load(p + 'Calls_tag_list.npy')

    # Convert matlab files to pyhton
    fs = 480 * 1000  # sampling of audio recordings
    calls, calls_names = mf.mattopy(stim_type, fs)

    # Tags and Stimulus names
    connection, c2 = mf.tagtostimulus(path_names)
    stimulus_tags = [''] * len(calls_names)
    for pp in range(len(calls_names)):
        s = calls_names[pp] + '.wav'
        stimulus_tags[pp] = connection[s]

    dur = np.array(duration) / 1000
    # if stim_length == 'single':
    #     dur = np.arange(0.01, 0.2, 0.01)
    # if stim_length == 'series':
    #     dur = np.arange(0.1, 1, 0.1)

    results = np.zeros(shape=(len(dur), 5))
    results_sync = np.zeros(shape=(len(dur), 5))
    results_dur = np.zeros(shape=(len(dur), 5))
    results_count = np.zeros(shape=(len(dur), 5))

    for j in tqdm(range(len(dur)), desc='Distances'):
        edges = [0, dur[j]]
        d = np.zeros(len(calls))
        d_sync = np.zeros(len(calls))
        d_dur = np.zeros(len(calls))
        d_count = np.zeros(len(calls))
        sp = [[]] * len(calls)
        for k in range(len(calls)):
            spike_times = [[]] * len(spikes[stimulus_tags[k]])
            for i in range(len(spikes[stimulus_tags[k]])):
                spike_times[i] = spk.SpikeTrain(np.sort(spikes[stimulus_tags[k]][i]), edges)
                # spike_times[i] = spk.SpikeTrain(list(spikes[stimulus_tags[k]][i]), edges)
            sp[k] = spike_times
            d[k] = abs(spk.isi_distance(spike_times, interval=[0, dur[j]]))
            d_sync[k] = spk.spike_sync(spike_times, interval=[0, dur[j]])
            # d[k] = abs(spk.spike_distance(spike_times, interval=[0, dur[j]]))
            # DUR metric
            last_spike = [[]] * len(spike_times)
            count = [[]] * len(spike_times)
            for kk in range(len(spike_times)):
                dummy = spike_times[kk][spike_times[kk] <= dur[j]]
                if len(dummy) > 0:
                    last_spike[kk] = np.max(dummy)
                    count[kk] = len(dummy)
                else:
                    last_spike[kk] = 0
                    count[kk] = 0
            d_dur[k] = np.mean(np.abs(np.diff(last_spike)))
            d_count[k] = np.mean(np.abs(np.diff(count)))

        sp = np.concatenate(sp)
        over_all = abs(spk.isi_distance(sp, interval=[0, dur[j]]))
        over_all_sync = spk.spike_sync(sp, interval=[0, dur[j]])
        last_spike = [[]] * len(sp)
        count = [[]] * len(sp)
        for kk in range(len(sp)):
            dummy = sp[kk][sp[kk] <= dur[j]]
            if len(dummy) > 0:
                last_spike[kk] = np.max(dummy)
                count[kk] = len(dummy)
            else:
                last_spike[kk] = 0
                count[kk] = 0
        over_all_dur = np.mean(np.abs(np.diff(last_spike)))
        over_all_count = np.mean(np.abs(np.diff(count)))

        ratio = over_all / np.mean(d)
        diff = over_all - np.mean(d)
        ratio_sync = over_all_sync / np.mean(d_sync)
        diff_sync = over_all_sync - np.mean(d_sync)
        ratio_dur = over_all_dur / np.mean(d_dur)
        diff_dur = over_all_dur - np.mean(d_dur)
        ratio_count = over_all_count / np.mean(d_count)
        diff_count = over_all_count - np.mean(d_count)

        results[j, :] = [np.mean(d), np.std(d), over_all, ratio, diff]
        results_sync[j, :] = [np.mean(d_sync), np.std(d_sync), over_all_sync, ratio_sync, diff_sync]
        results_dur[j, :] = [np.mean(d_dur), np.std(d_dur), over_all_dur, ratio_dur, diff_dur]
        results_count[j, :] = [np.mean(d_count), np.std(d_count), over_all_count, ratio_count, diff_count]

    # Save to HDD
    if extended:
        np.save(p + 'ISI_Ratios_' + stim_type + '_extended.npy', results)
        np.save(p + 'SYNC_Ratios_' + stim_type + '_extended.npy', results_sync)
        np.save(p + 'DUR_Ratios_' + stim_type + '_extended.npy', results_dur)
        np.save(p + 'COUNT_Ratios_' + stim_type + '_extended.npy', results_count)
    else:
        np.save(p + 'ISI_Ratios_' + stim_type + '.npy', results)
        np.save(p + 'SYNC_Ratios_' + stim_type + '.npy', results_sync)
        np.save(p + 'DUR_Ratios_' + stim_type + '.npy', results_dur)
        np.save(p + 'COUNT_Ratios_' + stim_type + '.npy', results_count)

    print('Ratios saved')

if PLOT_D_RATIOS:
    extended = False
    data_name = '2018-02-16-aa'
    # data_name = '2018-02-15-aa'
    # data_name = datasets[-1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    method = 'exp'
    p = path_names[1]
    # ==================================================================================================================
    # ISI and SYNC
    if extended:
        figname = path_names[2] + 'Distance_Ratios_ISI_SYNC_new_' + stim_type + '_extended.pdf'
        ratios_isi = np.load(p + 'ISI_Ratios_' + stim_type + '_extended.npy')
        ratios_sync = np.load(p + 'SYNC_Ratios_' + stim_type + '_extended.npy')
    else:
        figname = path_names[2] + 'Distance_Ratios_ISI_SYNC_new_' + stim_type + '.pdf'
        ratios_isi = np.load(p + 'ISI_Ratios_' + stim_type + '.npy')
        ratios_sync = np.load(p + 'SYNC_Ratios_' + stim_type + '.npy')

    # max_norm = np.max(ratios_isi[:, 2] * 1000)
    # ratios_isi[:, 0] = (ratios_isi[:, 0] * 1000) / max_norm
    # ratios_isi[:, 1] = (ratios_isi[:, 1] * 1000) / max_norm
    # ratios_isi[:, 2] = (ratios_isi[:, 2] * 1000) / max_norm
    #
    # max_norm = np.max(ratios_sync[:, 2] * 1000)
    # ratios_sync[:, 0] = (ratios_sync[:, 0] * 1000) / max_norm
    # ratios_sync[:, 1] = (ratios_sync[:, 1] * 1000) / max_norm
    # ratios_sync[:, 2] = (ratios_sync[:, 2] * 1000) / max_norm

    # Plot
    mf.plot_settings()
    if stim_length == 'series':
        x_end = 2500 + 100
        x_step = 500
    if stim_length == 'single':
        x_end = 250 + 10
        x_step = 50

    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    fig = plt.figure(figsize=(5.9, 2.9))
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1])
    ax3 = plt.subplot(grid[2])
    ax4 = plt.subplot(grid[3])

    ax1.errorbar(duration, ratios_isi[:, 0], yerr=ratios_isi[:, 1], color='k', marker='o', label='within')
    ax1.plot(duration, ratios_isi[:, 2], '-', label='between', color='blue')
    ax1.plot(duration, ratios_isi[:, 4], 'g-', label='diff')
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.set_xticklabels([])
    ax1.set_ylabel('ISI')
    ax1.set_xlim(0, x_end)
    ax1.set_xticks(np.arange(0, x_end, x_step))

    ax3.plot(duration, ratios_isi[:, 3], 'r-', label='ratio')

    ax3.set_ylim(1, 3)
    ax3.set_yticks(np.arange(1, 3.1, 0.5))
    ax3.set_ylabel('Ratio')
    ax3.set_xlim(0, x_end)
    ax3.set_xticks(np.arange(0, x_end, x_step))

    ax2.errorbar(duration, ratios_sync[:, 0], yerr=ratios_sync[:, 1], color='k', marker='o', label='within')
    ax2.plot(duration, ratios_sync[:, 2], '-', label='between', color='blue')
    ax2.plot(duration, abs(ratios_sync[:, 4]), 'g-', label='diff')
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(1, 1.1, 0.2))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_ylabel('SYNC')
    ax2.set_xlim(0, x_end)
    ax2.set_xticks(np.arange(0, x_end, x_step))

    ax4.plot(duration, 1/ratios_sync[:, 3], 'r-', label='ratio')
    ax4.set_ylim(1, 3)
    ax4.set_yticks(np.arange(1, 3.1, 0.5))
    ax4.set_yticklabels([])
    ax4.set_xlim(0, x_end)
    ax4.set_xticks(np.arange(0, x_end, x_step))

    # Axes Labels
    fig.text(0.5, 0.055, 'Spike train duration [ms]', ha='center', fontdict=None)

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                 color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                 color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')
    ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
             color='black')
    sns.despine()

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.4)
    # figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/Distance_Ratios_' + stim_type + '.pdf'
    fig.savefig(figname)
    plt.close(fig)

    # ==================================================================================================================
    # DUR and COUNT
    if extended:
        ratios_isi = np.load(p + 'DUR_Ratios_' + stim_type + '_extended.npy')
        ratios_sync = np.load(p + 'COUNT_Ratios_' + stim_type + '_extended.npy')
        figname = path_names[2] + 'Distance_Ratios_DUR_COUNT_new_' + stim_type + '_extended.pdf'
    else:
        figname = path_names[2] + 'Distance_Ratios_DUR_COUNT_new_' + stim_type + '.pdf'
        ratios_isi = np.load(p + 'DUR_Ratios_' + stim_type + '.npy')
        ratios_sync = np.load(p + 'COUNT_Ratios_' + stim_type + '.npy')

    # max_norm = np.max(ratios_isi[:, 2] * 1000)
    # ratios_isi[:, 0] = (ratios_isi[:, 0] * 1000) / max_norm
    # ratios_isi[:, 1] = (ratios_isi[:, 1] * 1000) / max_norm
    # ratios_isi[:, 2] = (ratios_isi[:, 2] * 1000) / max_norm
    #
    # max_norm = np.max(ratios_sync[:, 2] * 1000)
    # ratios_sync[:, 0] = (ratios_sync[:, 0] * 1000) / max_norm
    # ratios_sync[:, 1] = (ratios_sync[:, 1] * 1000) / max_norm
    # ratios_sync[:, 2] = (ratios_sync[:, 2] * 1000) / max_norm

    # Plot
    mf.plot_settings()
    if stim_length == 'series':
        x_end = 2500 + 100
        x_step = 500
        y_DUR_step = 0.05
        y_COUNT_step = 10
        y_COUNT = 40
        y_DUR = np.max(ratios_isi[:, 0] + ratios_isi[:, 1])
    if stim_length == 'single':
        x_end = 250 + 10
        x_step = 50
        y_DUR_step = 0.02
        y_DUR = 0.1
        y_COUNT_step = 5
        y_COUNT = 20

    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    fig = plt.figure(figsize=(5.9, 2.9))
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1])
    ax3 = plt.subplot(grid[2])
    ax4 = plt.subplot(grid[3])

    ax1.errorbar(duration, ratios_isi[:, 0], yerr=ratios_isi[:, 1], color='k', marker='o', label='within')
    ax1.plot(duration, ratios_isi[:, 2], '-', label='between', color='blue')
    ax1.plot(duration, abs(ratios_isi[:, 0]-ratios_isi[:, 2]), 'g-', label='diff')
    ax1.set_ylim(0, y_DUR)
    ax1.set_yticks(np.arange(0, y_DUR+y_DUR_step, y_DUR_step))
    ax1.set_xticklabels([])
    ax1.set_ylabel('DUR')
    ax1.set_xlim(0, x_end)
    ax1.set_xticks(np.arange(0, x_end, x_step))

    ax3.plot(duration, ratios_isi[:, 3], 'r-', label='ratio')

    ax3.set_ylim(1, 3)
    ax3.set_yticks(np.arange(1, 3.1, 0.5))
    ax3.set_ylabel('Ratio')
    ax3.set_xlim(0, x_end)
    ax3.set_xticks(np.arange(0, x_end, x_step))

    ax2.errorbar(duration, ratios_sync[:, 0], yerr=ratios_sync[:, 1], color='k', marker='o', label='within')
    ax2.plot(duration, ratios_sync[:, 2], '-', label='between', color='blue')
    ax2.plot(duration, abs(ratios_sync[:, 0]-ratios_sync[:, 2]), 'g-', label='diff')
    ax2.set_ylim(0, y_COUNT)
    ax2.set_yticks(np.arange(0, y_COUNT+y_COUNT_step, y_COUNT_step))
    # ax2.set_xticklabels([])
    # ax2.set_yticklabels([])
    ax2.set_ylabel('COUNT')
    ax2.set_xlim(0, x_end)
    ax2.set_xticks(np.arange(0, x_end, x_step))

    ax4.plot(duration, ratios_sync[:, 3], 'r-', label='ratio')
    ax4.set_ylim(1, 3)
    ax4.set_yticks(np.arange(1, 3.1, 0.5))
    ax4.set_yticklabels([])
    ax4.set_xlim(0, x_end)
    ax4.set_xticks(np.arange(0, x_end, x_step))

    # Axes Labels
    fig.text(0.5, 0.055, 'Spike train duration [ms]', ha='center', fontdict=None)

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')
    ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
             color='black')
    sns.despine()

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.4)
    # figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/Distance_Ratios_' + stim_type + '.pdf'
    fig.savefig(figname)
    plt.close(fig)

    # data_name = '2018-02-09-aa'
    # # data_name = '2018-02-15-aa'
    # # data_name = datasets[-1]
    # path_names = mf.get_directories(data_name=data_name)
    # print(data_name)
    # method = 'exp'
    # p = path_names[1]
    # ratios_isi = np.load(p + 'ISI_Ratios_' + stim_type + '.npy')
    # ratios_sync = np.load(p + 'SYNC_Ratios_' + stim_type + '.npy')
    #
    # # max_norm = np.max(ratios_isi[:, 2] * 1000)
    # # ratios_isi[:, 0] = (ratios_isi[:, 0] * 1000) / max_norm
    # # ratios_isi[:, 1] = (ratios_isi[:, 1] * 1000) / max_norm
    # # ratios_isi[:, 2] = (ratios_isi[:, 2] * 1000) / max_norm
    #
    # # Plot
    # mf.plot_settings()
    # if stim_length == 'series':
    #     x_end = 2500 + 100
    #     x_step = 500
    # if stim_length == 'single':
    #     x_end = 250 + 10
    #     x_step = 50
    #
    # # Create Grid
    # grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    # fig = plt.figure(figsize=(5.9, 2.9))
    # ax1 = plt.subplot(grid[0])
    # ax2 = plt.subplot(grid[1])
    # ax3 = plt.subplot(grid[2])
    # ax4 = plt.subplot(grid[3])
    #
    # ax1.errorbar(duration, ratios_isi[:, 0], yerr=ratios_isi[:, 1], color='k', marker='o', label='within')
    # ax1.plot(duration, ratios_isi[:, 2], '-', label='between', color='blue')
    # ax1.set_ylim(0, 1)
    # ax1.set_yticks(np.arange(0, 1.1, 0.2))
    # ax1.set_xticklabels([])
    # ax1.set_ylabel('ISI Distance')
    # ax1.set_xlim(0, x_end)
    # ax1.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax3.plot(duration, ratios_isi[:, 3], 'r-', label='ratio')
    # ax3.set_ylim(1, 3)
    # ax3.set_yticks(np.arange(1, 3.1, 0.5))
    # ax3.set_ylabel('Ratio')
    # ax3.set_xlim(0, x_end)
    # ax3.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax2.errorbar(duration, ratios_sync[:, 0], yerr=ratios_sync[:, 1], color='k', marker='o', label='within')
    # ax2.plot(duration, ratios_sync[:, 2], '-', label='between', color='blue')
    # ax2.set_ylim(0, 1)
    # ax2.set_yticks(np.arange(0, 1.1, 0.2))
    # ax2.set_xticklabels([])
    # ax2.set_yticklabels([])
    # ax2.set_ylabel('SYNC Value')
    # ax2.set_xlim(0, x_end)
    # ax2.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax4.plot(duration, 1 / ratios_sync[:, 3], 'r-', label='ratio')
    # ax4.set_ylim(1, 3)
    # ax4.set_yticks(np.arange(1, 3.1, 0.5))
    # ax4.set_yticklabels([])
    # ax4.set_xlim(0, x_end)
    # ax4.set_xticks(np.arange(0, x_end, x_step))
    #
    # # Axes Labels
    # fig.text(0.5, 0.055, 'Spike train duration [ms]', ha='center', fontdict=None)
    #
    # # Subplot caps
    # subfig_caps = 12
    # label_x_pos = 0.05
    # label_y_pos = 0.90
    # subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    # ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
    #          color='black')
    # ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
    #          color='black')
    # ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
    #          color='black')
    # ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
    #          color='black')
    # sns.despine()
    #
    # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.4)
    # figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/Distance_Ratios_' + stim_type + '.pdf'
    # # figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/Distance_Ratios_TEST_' + stim_type + '.pdf'
    # fig.savefig(figname)
    # plt.close(fig)

if CALL_STRUC:
    # Try to load e pulses from HDD
    data_name = '2018-02-09-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    method = 'exp'
    p = path_names[1]
    spikes = np.load(p + 'Calls_spikes.npy').item()
    tag_list = np.load(p + 'Calls_tag_list.npy')

    # Raster Plot
    # call_nr = 1
    for call_nr in range(4):  #range(len(calls)):
        plt.subplot(4, 1, call_nr+1)
        spike_times = spikes[stimulus_tags[call_nr]]
        y = np.arange(0, len(spike_times), 1)
        for k in range(5):  #range(len(spike_times)):
            plt.plot(spike_times[k], np.zeros(len(spike_times[k])) + k/4, 'k|')
        plt.plot(calls[call_nr], np.zeros(len(calls[call_nr])) + k/4 + 0.5, 'r|')
    plt.show()

    y = np.arange(0, len(calls), 1)
    labels = [[]] * len(calls)
    for k in range(len(calls)):
        plt.plot(calls[k], np.zeros(len(calls[k])) + y[k], 'k|')
        labels[k] = calls_names[k][17:]

    plt.yticks(y, labels, rotation='horizontal')
    plt.axvline(0.1, color='k')
    plt.axvline(0.5, color='k')
    plt.axvline(1, color='k')
    plt.axvline(2, color='k')
    plt.show()

if GAP:
    # dat = datasets[5]
    # dat = datasets[-2]
    old = False
    vs_order = 2
    protocol_name = 'Gap'
    spike_detection = True
    show_detection = False
    data_set_number = 11
    data_name = datasets[data_set_number]
    print(str(data_set_number) + ' of ' + str(len(datasets)))
    print(data_name)
    path_names = mf.get_directories(data_name=data_name)

    # tag_list = np.load(path_names[1] + 'Gap_tag_list.npy')
    if spike_detection:
        mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                           filter_on=True, window=None, mph_percent=mph_percent)

    mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=[True, True], save_data=False,
                         old=old, vs_order=vs_order)

if save_extended_spikes:
    path_names = mf.get_directories('2018-02-16-aa')
    sp = np.load(path_names[1] + 'Calls_spikes.npy').item()
    mf.extend_spike_train(sp, gap=[0.05, 0.01, 0.05, 0.01], extension_time=[4, 0.18, 4, 0.18], mode='all', path_names=path_names)
    print('Spike trains extended and saved')

if POISSON_TRAINS:
    rec = '2018-02-16-aa'
    path_names = mf.get_directories(rec)
    trials = 20
    rate = 100
    # tmax_range = [0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3]
    # tmax_range = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
    if stim_type is 'poisson_diff':
        tmax_range = np.round(np.linspace(0.05, 3, 20), 2)
    if stim_type is 'poisson_2diff':
        tmax_range = np.round(np.linspace(0.05, 3, 10), 2)
        tmax_range =np.sort(np.append(tmax_range, tmax_range))
    if stim_type is 'poisson_same':
        tmax_range = np.zeros(20) + 2

    spikes = {}
    spike_trains = {}
    ps = [[]] * len(tmax_range)
    ps_tag = [[]] * len(tmax_range)
    for k in range(len(tmax_range)):
        ps[k], _ = mf.poission_spikes(trials, rate, tmax_range[k])
        ps_tag[k] = 'p' + str(k)
        poisson_spike_trains = [[]] * len(ps[k])
        for i in range(len(ps[k])):
            poisson_spike_trains[i] = spk.SpikeTrain(ps[k][i], [0, tmax_range[k]])
        spike_trains.update({'p' + str(k): poisson_spike_trains})
        spikes.update({'p' + str(k): ps[k]})
    np.save(path_names[1] + stim_type + '_spikes.npy', spikes)
    np.save(path_names[1] + stim_type + '_spk_spike_trains.npy', spikes)
    np.save(path_names[1] + stim_type + '_tags.npy', ps_tag)
    print('Poisson Spike trains saved: ' + stim_type)

# Rect Intervals
if INTERVAL_REC_SPONT:
    spont = True
    # good recordings
    datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2018-02-16-aa']
    datasets = [datasets[0]]
    vs_order = 2
    old = False
    protocol_name = 'PulseIntervalsRect'
    spike_detection = True
    for dat in tqdm(range(len(datasets)), desc='Interval Rect'):
        print(str(dat) + ' of ' + str(len(datasets)))
        data_name = datasets[dat]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        if spike_detection:
            show_detection = True
            if spont:
                mf.spike_times_gap_spont(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                                   filter_on=True, window=None, mph_percent=mph_percent)
            else:
                mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=False,
                                         th_factor=th_factor,
                                         filter_on=True, window=None, mph_percent=mph_percent)

        # mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=[True, True], save_data=True,
        #                      old=old, vs_order=vs_order)
    # mf.plot_cohen(protocol_name, datasets, save_fig=True)

if INTERVAL_REC:
    # good recordings
    datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2018-02-16-aa']
    # datasets = ['2017-12-05-a']
    vs_order = 2
    old = False
    protocol_name = 'PulseIntervalsRect'
    spike_detection = False
    for dat in tqdm(range(len(datasets)), desc='Interval Rect'):
        print(str(dat) + ' of ' + str(len(datasets)))
        data_name = datasets[dat]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        if spike_detection:
            show_detection = False
            mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                               filter_on=True,
                               window=None, mph_percent=mph_percent)
        mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=False, show=[False, False], save_data=True,
                             old=old, vs_order=vs_order)
    # mf.plot_cohen(protocol_name, datasets, save_fig=True)

# Analyse Intervals MothASongs data stored on HDD
if INTERVAL_MAS:
    # good recordings:
    datasets = ['2017-11-27-aa', '2017-12-01-ab', '2017-12-01-ac', '2017-12-05-ab', '2017-11-29-aa']
    # datasets = ['2017-11-27-aa']
    old = False
    vs_order = 2
    protocol_name = 'intervals_mas'
    spike_detection = False
    show_detection = False
    for k in tqdm(range(len(datasets)), desc='Interval MAS'):
        data_name = datasets[k]
        # old 0:9
        print(data_name)
        if old:
            print('OLD MAS Protocol!')
        path_names = mf.get_directories(data_name=data_name)

        if spike_detection:
            mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                               filter_on=True, window=None, mph_percent=mph_percent)

        mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=[True, True], save_data=True, old=old, vs_order=vs_order)

    # OLD:
    # mf.moth_intervals_spike_detection(path_names, window=None, th_factor=th_factor, mph_percent=mph_percent,
    #                                   filter_on=True, save_data=False, show=show, bin_size=bin_size)
    # mf.moth_intervals_analysis(datasets[0])

if POISSON:
    vs_order = 1
    mode = 'all'
    pp = np.arange(0.5, 50, 0.5)

    if mode is 'nsamples':
        compute_pspikes = True
        plot_it = False
        nsamples = [1, 10, 100]
        info = nsamples
        xx = len(nsamples)
        rates = [100] * xx
        info2 = 'f=' + str(rates[0]) + '_Hz'
    if mode is 'rates':
        compute_pspikes = True
        plot_it = False
        rates = [50, 100, 200, 400]
        info = rates
        xx = len(rates)
        nsamples = [10] * xx
        info2 = 'n=' + str(nsamples[0])
    if mode is 'all':
        rates = np.array([50, 100, 200, 400])
        nsamples = np.array([1, 10, 100])
        compute_pspikes = False
        plot_it = True

    tmin = 0
    # ts = np.arange(0.005, 1, 0.005)
    ts = np.logspace(np.log(0.01), np.log(1), 400, base=np.exp(1))
    if compute_pspikes:
        # vs_tmax = np.zeros(shape=(len(rates), len(ts)))
        vs_tmax = np.zeros(shape=(xx, len(ts)))
        for jj in tqdm(range(xx)):
            for tt in range(len(ts)):
                p_spikes, isi_p = mf.poission_spikes(nsamples[jj], rates[jj], ts[tt])
                _, _, vs_mean_boot, _, vs_percentile_boot, vs_ci_boot = \
                    mf.vs_range(p_spikes, pp / 1000, tmin=tmin, n_ci=0, order=vs_order)
                vs_tmax[jj, tt] = np.nanmean(vs_mean_boot)
        np.save('/media/brehm/Data/MasterMoth/figs/' + 'vs_tmax_' + mode + '_order_' + str(vs_order) + '.npy', vs_tmax)
        print('Data Saved')
    else:
        # vs_tmax = np.load('/media/brehm/Data/MasterMoth/figs/' + 'vs_tmax_' + mode + '.npy')
        vs_rates = np.load('/media/brehm/Data/MasterMoth/figs/' + 'vs_tmax_' + 'rates' + '_order_' + str(vs_order) + '.npy')
        vs_samples = np.load('/media/brehm/Data/MasterMoth/figs/' + 'vs_tmax_' + 'nsamples' + '_order_' + str(vs_order) + '.npy')

    if plot_it:
        # Fit and Plot
        # mf.poisson_vs_duration(ts, vs_tmax, info, mode=mode)
        a = [True] * len(vs_rates)
        # a[1] = False
        mf.poisson_vs_duration2(ts, vs_rates[a], vs_samples, rates[a], nsamples)

        # Save Plot to HDD
        # figname = '/media/brehm/Data/MasterMoth/figs/' + 'Poisson_tmax_VS_' + info2 + '.pdf'
        figname = '/media/brehm/Data/MasterMoth/figs/PoissonDuration_VS_order_' + str(vs_order) + '.pdf'
        fig = plt.gcf()
        fig.set_size_inches(5.9, 2.9)
        fig.savefig(figname)
        plt.close(fig)

# Analyse FIField data stored on HDD
if FIFIELD:
    save_analysis = False
    species = 'Carales'
    durs = 20

    if species is 'Estigmene':
        # Estigmene:
        if durs == 20:
            datasets = ['2017-11-25-aa', '2017-11-27-aa']  # 20 ms
        if durs == 50:
            datasets = ['2017-10-26-aa', '2017-12-05-aa']  # 50 ms
        # datasets = ['2017-10-26-aa', '2017-11-25-aa', '2017-11-27-aa', '2017-12-05-aa']
    elif species is 'Carales':
        # Carales:
        if durs == 20:
            datasets = ['2017-11-01-aa', '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']  # 20 ms
        if durs == 50:
            datasets = ['2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac']  # 50 ms
        if durs == 5:
            datasets = ['2017-10-23-ah']  # 5 ms
        # datasets = ['2017-10-23-ah', '2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac', '2017-11-01-aa',
        #             '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']

    print(species + ': ' + str(durs) + ' ms')
    fi = [[]] * len(datasets)
    duration = []
    for k in range(len(datasets)):
        data_set_number = k
        data_name = datasets[data_set_number]
        print(str(data_set_number+1) + ' of ' + str(len(datasets)))
        print(data_name)
        path_names = mf.get_directories(data_name=data_name)

        plot_fi_field = True
        plot_fi_curves = True
        data = path_names[0]
        p = path_names[1]
        ths = [100, 125, 150, 175, 200]
        th = 150
        if save_analysis:
            spike_count, rate, fsl, fisi, d_isi, instant_rate, conv_rate, dur = mf.fifield_analysis2(path_names)
            all_data = [spike_count, rate, fsl, fisi, d_isi, instant_rate, conv_rate, dur]
            with open(p + 'fi_analysis.txt', 'wb') as fp:  # Pickling
                pickle.dump(all_data, fp)
            continue
        else:
            with open(p + 'fi_analysis.txt', 'rb') as fp:  # Unpickling
                all_data = pickle.load(fp)
                spike_count, rate, fsl, fisi, d_isi, instant_rate, conv_rate, dur = all_data

        duration.append(dur)
        freqs = [[]] * len(rate)
        i = 0
        for key in rate:
            freqs[i] = int(key)
            i += 1
        freqs = sorted(freqs)

        estimated_th_d = np.zeros(len(freqs))
        estimated_th_r = np.zeros(len(freqs))
        estimated_th_inst = np.zeros(len(freqs))
        estimated_th_conv = np.zeros(len(freqs))
        # data_th_d = np.zeros(len(freqs))
        # data_th_r = np.zeros(len(freqs))
        i = -1
        for ff in tqdm(freqs, desc='FI Curves'):
            i += 1
            # Fit Boltzman
            x_d, y_d, params_d, perr_d, y0_d = mf.fit_function(d_isi[ff][:, 0], d_isi[ff][:, 2])
            x_r, y_r, params_r, perr_r, y0_r = mf.fit_function(spike_count[ff][:, 0], spike_count[ff][:, 1])
            # x_inst, y_inst, params_inst, perr_inst, y0_inst = mf.fit_function(instant_rate[ff][:, 0], instant_rate[ff][:, 1])
            x_conv, y_conv, params_conv, perr_conv, y0_conv = mf.fit_function(conv_rate[ff][:, 0], conv_rate[ff][:, 1])

            # Compute Fitting Error
            # print(str(ff) + ' kHz: Summed Error = ' + str(perr_conv[2]+perr_conv[1]))
            k_conv = params_conv[-1]
            k_d = params_d[-1]
            # k_inst = params_inst[-1]
            k_r = params_r[-1]

            I0_d = params_d[-2]
            I0_r = params_r[-2]
            # I0_inst = params_inst[-2]
            I0_conv = params_conv[-2]

            slope_d = (params_d[1]-params_d[0])*k_d / 4
            slope_r = (params_r[1]-params_r[0])*k_r / 4
            # slope_inst = (params_inst[1]-params_inst[0])*k_inst / 4
            slope_conv = (params_conv[1]-params_conv[0])*k_conv / 4

            c_d = y0_d / slope_d - I0_d
            c_r = y0_r / slope_r - I0_r
            # c_inst = y0_inst / slope_inst - I0_inst
            c_conv = y0_conv / slope_conv - I0_conv


            summed_error_conv = perr_conv[1]
            summed_error_d = perr_d[1]
            summed_error_r = perr_r[1]
            # summed_error_inst = perr_inst[1]
            top_bottom_d = params_d[1] - params_d[0]

            # if ff == 5:
            #     embed()
            #     exit()
            # Estimate Threshold directly from Data
            # threshold_d = 0.5
            # threshold_r = 4
            # try:
            #     idx_d = y_d >= threshold_d
            #     th_d = x_d[idx_d][0]
            #     idx_r = y_r >= threshold_r
            #     th_r = x_r[idx_r][0]
            # except:
            #     th_d = np.nan
            #     th_r = np.nan
            #     print(str(ff) + ' kHz: Could not find threshold from data')
            # data_th_d[i] = th_d
            # data_th_r[i] = th_r

            # Estimate Thresholds from Boltzman Fit
            th_d_fit = I0_d - (2/k_d)
            th_r_fit = I0_r - (2/k_r)
            # th_inst_fit = I0_inst - (2/k_inst)
            th_conv_fit = I0_conv - (2/k_conv)

            # Linear approximation
            d_approx = slope_d * (x_d + c_d)
            r_approx = slope_r * (x_r + c_r)
            # inst_approx = slope_inst * (x_inst + c_inst)
            conv_approx = slope_conv * (x_conv + c_conv)

            # # V50
            # th_d_fit = params_d[2]
            # th_r_fit = params_r[2]
            # th_inst_fit = params_inst[2]
            # th_conv_fit = params_conv[2]

            limit_d = 0.4
            limit_r = np.min(y_r) + 2
            limit_conv = np.min(y_conv) * 1.5
            # limit_inst = np.min(y_inst) * 1.5

            no_th = [False] * 4
            control_for_no_response = True
            if control_for_no_response:
                if np.max(y_d) < limit_d or np.max(y_d) <= 0 or slope_d <= 0:
                    estimated_th_d[i] = np.max(d_isi[ff][:, 0])
                    no_th[0] = True
                else:
                    estimated_th_d[i] = th_d_fit

                if np.max(y_r) < limit_r or np.max(y_r) <= 0 or slope_r <= 0:
                    estimated_th_r[i] = np.max(spike_count[ff][:, 0])
                    no_th[1] = True
                else:
                    estimated_th_r[i] = th_r_fit

                # if np.max(y_inst) < limit_inst or np.max(y_inst) <= 0:
                #     estimated_th_inst[i] = np.max(instant_rate[ff][:, 0])
                #     no_th[2] = True
                # else:
                #     estimated_th_inst[i] = th_inst_fit

                if np.max(y_conv) < limit_conv or np.max(y_conv) <= 0 or slope_conv <= 0:
                    estimated_th_conv[i] = np.max(conv_rate[ff][:, 0])
                    no_th[3] = True
                else:
                    estimated_th_conv[i] = th_conv_fit
            else:
                estimated_th_d[i] = th_d_fit
                estimated_th_r[i] = th_r_fit
                estimated_th_conv[i] = th_conv_fit

            # if ff == 5:
            #     embed()
            #     exit()

            # Plot FI Curves
            x_min = 10
            x_max = 90
            x_step = 10
            subfig_caps = 12
            mf.plot_settings()
            if plot_fi_curves:
                label_x_pos = -0.2
                label_y_pos = 1.1
                fig = plt.figure()
                
                ax1 = plt.subplot(2, 2, 3)
                ax1.plot(d_isi[ff][:, 0], d_isi[ff][:, 2], 'ko', label='sync')
                ax1.plot(x_d, y_d, 'k')
                # ax1.plot([th_d_fit, th_d_fit], [0, 1], 'k--')
                ax1.plot(x_d, d_approx, 'k--')
                ax1.set_ylabel('SYNC value')
                ax1.set_ylim(0, 1)
                ax1.text(label_x_pos, label_y_pos, 'c', transform=ax1.transAxes, size=subfig_caps)
                ax1.set_xlim(x_min, x_max)
                ax1.set_xticks(np.arange(x_min, x_max+x_step, x_step))

                ax2 = plt.subplot(2, 2, 2)
                y_max = 12
                ax2.errorbar(spike_count[ff][:, 0], spike_count[ff][:, 1], yerr=spike_count[ff][:, 2], marker='o', linestyle='', color='k', label='spike_count')
                ax2.plot(x_r, y_r, 'k')
                # ax2.plot([th_r_fit, th_r_fit], [0, y_max], 'k--')
                ax2.plot(x_r, r_approx, 'k--')
                ax2.set_ylabel('Spike count')
                ax2.set_ylim(0, y_max)
                ax2.set_yticks(np.arange(0, y_max+2, 2))
                ax2.text(label_x_pos, label_y_pos, 'b', transform=ax2.transAxes, size=subfig_caps)
                ax2.set_xlim(x_min, x_max)
                ax2.set_xticks(np.arange(x_min, x_max + x_step, x_step))

                ax3 = plt.subplot(2, 2, 4)
                y_max = 30
                ax3.errorbar(fsl[ff][:, 0], fsl[ff][:, 1], yerr=fsl[ff][:, 2], marker='o',
                             linestyle='-', color='k', label='first spike latency')
                ax3.errorbar(fisi[ff][:, 0], fisi[ff][:, 1], yerr=fisi[ff][:, 2], marker='s',
                             linestyle='-', color='0.3', label='first spike interval')
                ax3.legend(loc='best',frameon=False)
                ax3.set_ylabel('Time [ms]')
                ax3.set_ylim(0, y_max)
                ax3.set_yticks(np.arange(0, y_max+5, 5))
                ax3.text(label_x_pos, label_y_pos, 'd', transform=ax3.transAxes, size=subfig_caps)
                ax3.set_xlim(x_min, x_max)
                ax3.set_xticks(np.arange(x_min, x_max + x_step, x_step))

                ax4 = plt.subplot(2, 2, 1)
                y_max = 600
                ax4.errorbar(conv_rate[ff][:, 0], conv_rate[ff][:, 1], yerr=conv_rate[ff][:, 2], marker='o', linestyle='', color='k', label='firing rate')
                ax4.plot(x_conv, y_conv, 'k')
                # ax4.plot([th_conv_fit, th_conv_fit], [0, y_max], 'k--')
                ax4.plot(x_conv, conv_approx, 'k--')
                # ax4.errorbar(instant_rate[ff][:, 0], instant_rate[ff][:, 1], yerr=instant_rate[ff][:, 2], marker='s', linestyle='', color='0.25', label='instantaneous rate')
                # ax4.plot(x_inst, y_inst, '0.25')
                # ax4.plot([th_inst_fit, th_inst_fit], [0, np.max(instant_rate[ff][:, 1])], '--', color='0.25')
                ax4.set_ylabel('Firing rate [Hz]')
                ax4.set_ylim(0, y_max)
                ax4.set_yticks(np.arange(0, y_max+100, 100))
                # plt.legend(frameon=False)
                ax4.text(label_x_pos, label_y_pos, 'a', transform=ax4.transAxes, size=subfig_caps)
                ax4.set_xlim(x_min, x_max)
                ax4.set_xticks(np.arange(x_min, x_max + x_step, x_step))

                sns.despine()
                fig.text(0.5, 0.035, 'Intensity [dB SPL]', ha='center', fontdict=None)
                fig.set_size_inches(5.9, 5.9)
                fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.4, hspace=0.4)
                figname = p + 'FICRUVE_' + str(ff) + '.pdf'
                fig.savefig(figname)
                plt.close(fig)

        if plot_fi_field:
            mf.plot_settings()
            fig, ax = plt.subplots()
            ax.plot(freqs, estimated_th_d, 'k-.', label='SYNC')
            ax.plot(freqs, estimated_th_r, 'k--', label='spike count')
            ax.plot(freqs, estimated_th_inst, 'k:', label='instantaneous rate')
            ax.plot(freqs, estimated_th_conv, '-', label='convolved rate', color='0.5')
            # plt.plot(freqs, data_th_d, 'b:', alpha=0.5, label='Data SYNC')
            # plt.plot(freqs, data_th_r, 'k:', alpha=0.5, label='Data spike count')
            ax.legend(frameon=False)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel('Intensity [dB SPL]')
            ax.set_xlim(0, 110)
            ax.set_xticks(np.arange(0, 110, 10))
            ax.set_ylim(10, 100)
            ax.set_yticks(np.arange(10, 100, 10))
            sns.despine()

            # fig = plt.gcf()
            fig.set_size_inches(2.9, 1.9)
            fig.subplots_adjust(left=0.1, top=0.8, bottom=0.2, right=0.9, wspace=0.5, hspace=0.1)
            # plt.show()
            figname = p + 'FIFIELD.pdf'
            fig.savefig(figname)
            plt.close(fig)

        # Save Data to HDD
        all_data = [freqs, estimated_th_conv, estimated_th_d, estimated_th_r, estimated_th_inst]
        with open(p + 'fi_field.txt', 'wb') as fp:  # Pickling
            pickle.dump(all_data, fp)
        print('FIField Data saved')

if Bootstrapping:
    # mf.resampling(datasets)
    nresamples = 10000
    mf.bootstrapping_vs(path_names, nresamples, plot_histogram=True)

if SOUND:  # Stimuli = Calls
    spikes = mf.spike_times_calls(path_names, 'Calls', show=False, save_data=True, th_factor=4, filter_on=True,
                                  window=None)

if EPULSES:
    datasets = ['2018-02-16-aa']
    # datasets = [datasets[-1]]
    for i in range(len(datasets)):
        data_name = datasets[i]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        method = 'exp'
        r = Parallel(n_jobs=-2)(delayed(mf.trains_to_e_pulses)(path_names, taus[k] / 1000, dt, stim_type=stim_type
                                                               , method=method, extended=extended) for k in range(len(taus)))
        print('Converting done')

if VANROSSUM:
    # Try to load e pulses from HDD
    # data_name = '2018-02-09-aa'
    # data_name = datasets[-1]
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    method = 'exp'
    p = path_names[1]

    # Compute VanRossum Distances
    correct = np.zeros((len(duration), len(taus)))
    dist_profs = {}
    matches = {}
    groups = {}
    for tt in tqdm(range(len(taus)), desc='taus', leave=False):
        try:
            # Load e-pulses if available:
            if extended:
                trains = np.load(p + 'epulses/' + 'e_trains_' + str(taus[tt]) + '_' + stim_type + '_extended.npy').item()
                stimulus_tags = np.load(p + 'epulses/' + 'stimulus_tags_' + str(taus[tt]) + '_' + stim_type + '_extended.npy')
            else:
                trains = np.load(p + 'epulses/' + 'e_trains_' + str(taus[tt]) + '_' + stim_type + '.npy').item()
                stimulus_tags = np.load(p + 'epulses/' + 'stimulus_tags_' + str(taus[tt]) + '_' + stim_type + '.npy')

            # print('Loading e-pulses from HDD done')
        except FileNotFoundError:
            # Compute e pulses if not available
            print('Could not find e-pulses!')

        distances = [[]] * len(duration)
        mm = [[]] * len(duration)
        gg = [[]] * len(duration)
        # Parallel loop through all durations for a given tau
        r = Parallel(n_jobs=-2)(delayed(mf.vanrossum_matrix)
                                (data_name, trains, stimulus_tags, duration[dur]/1000, dt, taus[tt]/1000,
                                 boot_sample=nsamples, save_fig=False) for dur in range(len(duration)))
        # mf.vanrossum_matrix2(data_name, trains, stimulus_tags, duration/1000, dt_factor, taus[tt]/1000, boot_
        # sample=nsamples, save_fig=True)
        # mm_mean, correct_matches, distances_per_boot, gg_mean = mf.vanrossum_matrix(data_name, trains, stimulus_tags,
        # duration[-1]/1000, dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=True)

        # Put values from parallel loop into correct variables
        for q in range(len(duration)):
            mm[q] = r[q][0]
            gg[q] = r[q][3]
            correct[q, tt] = r[q][1]
            distances[q] = r[q][2]
        dist_profs.update({taus[tt]: distances})
        matches.update({taus[tt]: mm})
        groups.update({taus[tt]: gg})  # not useful anymore
    # Save to HDD
    if extended:
        np.save(p + 'VanRossum_' + stim_type + '_extended.npy', dist_profs)
        np.save(p + 'VanRossum_correct_' + stim_type + '_extended.npy', correct)
        np.save(p + 'VanRossum_matches_' + stim_type + '_extended.npy', matches)
        np.save(p + 'VanRossum_groups_' + stim_type + '_extended.npy', groups)  # not useful anymore
    else:
        np.save(p + 'VanRossum_' + stim_type + '.npy', dist_profs)
        np.save(p + 'VanRossum_correct_' + stim_type + '.npy', correct)
        np.save(p + 'VanRossum_matches_' + stim_type + '.npy', matches)
        np.save(p + 'VanRossum_groups_' + stim_type + '.npy', groups)  # not useful anymore
    print('VanRossum Distances done')

if MVSB:
    extended = True
    # method = 'VanRossum'
    stim_type = 'all_series'
    stim_length = stim_type[4:]
    print(stim_length)
    print('extended: ' + str(extended))

    if stim_length == 'single':
        boarder = 20
    if stim_length == 'series':
        boarder = 17
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]

    if extended:
        matches = np.load(p + 'VanRossum_matches_' + stim_type + '_extended.npy').item()
    else:
        matches = np.load(p + 'VanRossum_matches_' + stim_type + '.npy').item()

    d_prime = np.zeros(shape=(len(taus), len(duration)))
    p_correct = np.zeros(shape=(len(taus), len(duration)))
    hit_rate = np.zeros(shape=(len(taus), len(duration)))
    fa_rate = np.zeros(shape=(len(taus), len(duration)))
    criterion_c = np.zeros(shape=(len(taus), len(duration)))
    percent_moth = [[]] * len(taus)
    for t in range(len(taus)):
        p_moths = [[]] * len(duration)
        for i in range(len(duration)):
            pm = np.zeros(len(matches[taus[t]][i]))
            hit = 0
            misses = 0
            false_alarms = 0
            correct_rejection = 0
            for k in range(len(matches[taus[t]][i])):
                m = sum(matches[taus[t]][i][:, k][0:boarder])
                b = sum(matches[taus[t]][i][:, k][boarder:])

                if k <= boarder:
                    a = 'noise'  # Stim is truely Moth
                    false_alarms += b
                    correct_rejection += m
                else:
                    a = 'signal'  # Stim is truely Bat
                    hit += b
                    misses += m
                pm[k] = m / (b+m)
            p_moths[i] = pm
            p_hit = hit / (hit+misses)
            p_false = false_alarms / (false_alarms+correct_rejection)
            p_correct[t, i] = 0.5 + (p_hit - p_false)/2
            out = mf.dPrime(hit, misses, false_alarms, correct_rejection)
            d_prime[t, i] = out['d']
            criterion_c[t, i] = out['c']
            hit_rate[t, i] = out['hit_rate']
            fa_rate[t, i] = out['fa_rate']
        percent_moth[t] = p_moths

    p_taus = [[]] * len(percent_moth)
    for j in range(len(percent_moth)):
        p_taus[j] = percent_moth[j][50]

    data = {}
    data = {'dprime': d_prime, 'c': criterion_c, 'hitrate': hit_rate, 'farate': fa_rate, 'pmoth': percent_moth,
            'pcorrect': p_correct, 'ptaus': p_taus}
    if extended:
        np.save(path_names[1] + 'MvsB_VanRossum_' + stim_length + '_extended.npy', data)
    else:
        np.save(path_names[1] + 'MvsB_VanRossum_' + stim_length + '.npy', data)

    print('MvsB VanRossum data saved')

    # fig = plt.figure()
    # plt.pcolormesh(d_prime, cmap='jet', vmin=0, vmax=2)
    # plt.colorbar()
    # fig.savefig(figname)
    # plt.close()
    #
    # fig = plt.figure()
    # plt.pcolormesh(percent_moth[9], cmap='jet', vmin=0, vmax=1)
    # plt.colorbar()
    # fig.savefig(figname2)
    # plt.close()
    #
    # fig = plt.figure()
    # plt.pcolormesh(p_taus, cmap='jet', vmin=0, vmax=1)
    # plt.colorbar()
    # fig.savefig(figname3)
    # plt.close()
    # print('Plot saved')

    # ==================================================================================================================
    # DISTANCES ========================================================================================================

    if extended:
        matches = np.load(p + 'distances_matches_' + stim_type + '_extended.npy').item()
        figname = path_names[2] + 'MvsB_distances_dprime_' + stim_type + '_extended.png'
    else:
        matches = np.load(p + 'distances_matches_' + stim_type + '.npy').item()
        figname = path_names[2] + 'MvsB_distances_dprime_' + stim_type + '.png'

    profiles = ['ISI', 'SYNC', 'DUR', 'COUNT']
    d_prime = np.zeros(shape=(len(taus), len(duration)))
    p_correct = np.zeros(shape=(len(taus), len(duration)))
    hit_rate = np.zeros(shape=(len(taus), len(duration)))
    fa_rate = np.zeros(shape=(len(taus), len(duration)))
    criterion_c = np.zeros(shape=(len(taus), len(duration)))
    percent_moth = [[]] * len(taus)
    for t in range(len(profiles)):
        p_moths = [[]] * len(duration)
        for i in range(len(duration)):
            pm = np.zeros(len(matches[profiles[t]][i]))
            hit = 0
            misses = 0
            false_alarms = 0
            correct_rejection = 0
            for k in range(len(matches[profiles[t]][i])):
                m = sum(matches[profiles[t]][i][:, k][0:boarder])
                b = sum(matches[profiles[t]][i][:, k][boarder:])

                if k <= boarder:
                    a = 'noise'  # Stim is truely Moth
                    false_alarms += b
                    correct_rejection += m
                else:
                    a = 'signal'  # Stim is truely Bat
                    hit += b
                    misses += m
                pm[k] = m / (b + m)
            p_moths[i] = pm
            p_hit = hit / (hit + misses)
            p_false = false_alarms / (false_alarms + correct_rejection)
            p_correct[t, i] = 0.5 + (p_hit - p_false) / 2
            out = mf.dPrime(hit, misses, false_alarms, correct_rejection)
            d_prime[t, i] = out['d']
            criterion_c[t, i] = out['c']
            hit_rate[t, i] = out['hit_rate']
            fa_rate[t, i] = out['fa_rate']
        percent_moth[t] = p_moths

    # Save Data
    data2 = {}
    data2 = {'dprime': d_prime, 'c': criterion_c, 'hitrate': hit_rate, 'farate': fa_rate, 'pmoth': percent_moth,
            'pcorrect': p_correct, 'ptaus': p_taus}
    if extended:
        np.save(path_names[1] + 'MvsB_Distances_' + stim_length + '_extended.npy', data2)
    else:
        np.save(path_names[1] + 'MvsB_Distances_' + stim_length + '.npy', data2)

    print('MvsB Distances data saved')

    # plt.figure()
    # for j in range(len(profiles)):
    #     plt.plot(duration, d_prime[j], label=profiles[j])
    # plt.ylim(0, 2)
    # plt.legend()
    # plt.savefig(figname)
    # plt.close()
    #
    # for j in range(len(profiles)):
    #     plt.figure()
    #     plt.pcolormesh(percent_moth[j], cmap='jet', vmin=0, vmax=1)
    #     plt.colorbar()
    #     if extended:
    #         plt.savefig(path_names[2] + 'MvsB_distances_percent_' + stim_type + '_' + profiles[j] + '_extended.png')
    #     else:
    #         plt.savefig(path_names[2] + 'MvsB_distances_percent_' + stim_type + '_' + profiles[j] + '.png')
    #     plt.close()
    # print('Plot saved')

if PLOT_MVSB:
    # method = 'VanRossum'
    # stim_type = 'all_single'
    # stim_length = stim_type[4:]

    boarder_series = 17
    boarder_single = 20
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    # Get Data
    data_vr_series_ext = np.load(path_names[1] + 'MvsB_VanRossum_' + 'series' + '_extended.npy').item()
    data_vr_single_ext = np.load(path_names[1] + 'MvsB_VanRossum_' + 'single' + '_extended.npy').item()
    data_d_series_ext = np.load(path_names[1] + 'MvsB_Distances_' + 'series' + '_extended.npy').item()
    data_d_single_ext = np.load(path_names[1] + 'MvsB_Distances_' + 'single' + '_extended.npy').item()
    data_vr_series = np.load(path_names[1] + 'MvsB_VanRossum_' + 'series' + '.npy').item()
    data_vr_single = np.load(path_names[1] + 'MvsB_VanRossum_' + 'single' + '.npy').item()
    data_d_series = np.load(path_names[1] + 'MvsB_Distances_' + 'series' + '.npy').item()
    data_d_single = np.load(path_names[1] + 'MvsB_Distances_' + 'single' + '.npy').item()

    # Plot percent moth
    mf.plot_settings()
    ax = [[]] * 21
    grid = matplotlib.gridspec.GridSpec(nrows=67, ncols=66)
    fig = plt.figure(figsize=(5.9, 5.9))
    ax[0] = plt.subplot(grid[0:10, 0:10])
    ax[1] = plt.subplot(grid[0:10, 14:24])

    ax[2] = plt.subplot(grid[0:10, 38:48])
    ax[3] = plt.subplot(grid[0:10, 52:62])

    ax[4] = plt.subplot(grid[14:24, 0:10])
    ax[5] = plt.subplot(grid[14:24, 14:24])

    ax[6] = plt.subplot(grid[14:24, 38:48])
    ax[7] = plt.subplot(grid[14:24, 52:62])

    ax[8] = plt.subplot(grid[28:38, 0:10])
    ax[9] = plt.subplot(grid[28:38, 14:24])

    ax[10] = plt.subplot(grid[28:38, 38:48])
    ax[11] = plt.subplot(grid[28:38, 52:62])

    ax[12] = plt.subplot(grid[42:52, 0:10])
    ax[13] = plt.subplot(grid[42:52, 14:24])

    ax[14] = plt.subplot(grid[42:52, 38:48])
    ax[15] = plt.subplot(grid[42:52, 52:62])

    ax[16] = plt.subplot(grid[56:66, 0:10])
    ax[17] = plt.subplot(grid[56:66, 14:24])

    ax[18] = plt.subplot(grid[56:66, 38:48])
    ax[19] = plt.subplot(grid[56:66, 52:62])

    # Colorbar
    ax[20] = plt.subplot(grid[0:66, 64:65])

    # Image Grid
    x_series = np.arange(0, 28, 1)
    y_series = np.arange(0, 2550, 50)
    y_series[0] = 10
    X_series, Y_series = np.meshgrid(x_series, y_series)

    x_single = np.arange(0, 32, 1)
    y_single = np.arange(0, 255, 5)
    y_single[0] = 1
    X_single, Y_single = np.meshgrid(x_single, y_single)
    tau_idx = 4

    # Subplot caps
    subfig_caps = 12
    label_x_pos = -0.4
    label_y_pos = 1.05
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    color_map = 'gray'
    ax[0].pcolormesh(X_series, Y_series, data_vr_series['pmoth'][tau_idx], cmap=color_map, vmin=0, vmax=1, rasterized=True)
    ax[1].pcolormesh(X_series, Y_series, data_vr_series_ext['pmoth'][tau_idx], cmap=color_map, vmin=0, vmax=1, rasterized=True)
    ax[2].pcolormesh(X_single, Y_single, data_vr_single['pmoth'][tau_idx], cmap=color_map, vmin=0, vmax=1, rasterized=True)
    ax[3].pcolormesh(X_single, Y_single, data_vr_single_ext['pmoth'][tau_idx], cmap=color_map, vmin=0, vmax=1, rasterized=True)

    k = 0
    for i in [4, 8, 12, 16]:
        ax[i].pcolormesh(X_series, Y_series, data_d_series['pmoth'][k], cmap=color_map, vmin=0, vmax=1, rasterized=True)
        ax[i+1].pcolormesh(X_series, Y_series, data_d_series_ext['pmoth'][k], cmap=color_map, vmin=0, vmax=1, rasterized=True)
        ax[i+2].pcolormesh(X_single, Y_single, data_d_single['pmoth'][k], cmap=color_map, vmin=0, vmax=1, rasterized=True)
        ax[i+3].pcolormesh(X_single, Y_single, data_d_single_ext['pmoth'][k], cmap=color_map, vmin=0, vmax=1, rasterized=True)
        k += 1

    # Axes ticks
    j = 0
    for i in [0, 4, 8, 12, 16]:  # series
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[j], transform=ax[i].transAxes, size=subfig_caps,
                   color='black')
        j += 2
        for k in range(2):
            ax[i+k].set_xticks(np.arange(0.5, 27.5, 10))
            ax[i+k].set_xticklabels([])
            ax[i+k].plot([boarder_series-0.25, boarder_series-0.25], [10, 2500], 'r-', lw=0.75)
            ax[i+k].set_yticks([0, 1000, 2000])
            ax[i+k].set_yticklabels([0, 1, 2])
            ax[i+k].annotate("", xy=(16.75, 2400), xycoords='data', xytext=(16.75, 3000), textcoords='data',
                           arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color='r', linewidth=0.75))
        ax[i+1].set_yticklabels([])
    j = 1
    for i in [2, 6, 10, 14, 18]:  # single
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[j], transform=ax[i].transAxes, size=subfig_caps,
                   color='black')
        j += 2
        for k in range(2):
            ax[i+k].set_xticks(np.arange(0.5, 31.5, 10))
            ax[i+k].set_xticklabels([])
            ax[i + k].plot([boarder_single-0.25, boarder_single-0.25], [1, 250], 'r-', lw=0.75)
            ax[i + k].set_yticks([0, 100, 200])
            ax[i + k].set_yticklabels([0, 0.1, 0.2])
            # path_effects = [path_effects.Stroke(linewidth=1, foreground='red'), path_effects.Normal()]
            ax[i+k].annotate("", xy=(19.75, 240), xycoords='data', xytext=(19.75, 300), textcoords='data',
                           arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color='r', linewidth=0.75))
        ax[i + 1].set_yticklabels([])

    ax[16].set_xticklabels(np.arange(0, 27, 10))
    ax[17].set_xticklabels(np.arange(0, 27, 10))
    ax[18].set_xticklabels(np.arange(0, 31, 10))
    ax[19].set_xticklabels(np.arange(0, 31, 10))

    # for i in range(len(ax)-1):
    #     ax[i].set_aspect('equal')

    for i in range(len(ax)-1):
        ax[i].text(0.18, 1.02, 'moths', transform=ax[i].transAxes, size=6, color='red')
        ax[i].text(0.7, 1.02, 'bats', transform=ax[i].transAxes, size=6, color='red')

    for i in range(4):
        ax[i].text(0.05, 0.02, r'$\tau$ = 10 ms', transform=ax[i].transAxes, size=6, color='k',
                   path_effects=[path_effects.Stroke(linewidth=1, foreground='w'), path_effects.Normal()])

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax[20], cmap=color_map, norm=norm)

    sz_text = 12
    ax[0].text(13, 3500, 'Call series', ha='center', va='center', fontdict=None, size=sz_text)
    ax[1].text(13, 3500, 'Cs extended', ha='center', va='center', fontdict=None, size=sz_text)
    ax[2].text(15, 350, 'Single calls', ha='center', va='center', fontdict=None, size=sz_text)
    ax[3].text(15, 350, 'Sc extended', ha='center', va='center', fontdict=None, size=sz_text)
    ax[20].text(8, 0.5, 'Percentage of moth classification', ha='center', va='center', fontdict=None, rotation=-90)
    fig.text(0.5, 0.05, 'Call number', ha='center', fontdict=None)
    fig.text(0.05, 0.5, 'Spike train duration [s]', ha='center', va='center', fontdict=None, rotation=90)

    sz_text = 8
    dist_x = -25
    ax[2].text(dist_x, 125, 'Van \nRossum', ha='center', va='center', fontdict=None, size=sz_text, color='k', bbox=dict(boxstyle='round', facecolor='w'))
    ax[6].text(dist_x, 125, 'ISI', ha='center', va='center', fontdict=None, size=sz_text, color='k', bbox=dict(boxstyle='round', facecolor='w'))
    ax[10].text(dist_x, 125, 'SYNC', ha='center', va='center', fontdict=None, size=sz_text, color='k', bbox=dict(boxstyle='round', facecolor='w'))
    ax[14].text(dist_x, 125, 'DUR', ha='center', va='center', fontdict=None, size=sz_text, color='k', bbox=dict(boxstyle='round', facecolor='w'))
    ax[18].text(dist_x, 125, 'COUNT', ha='center', va='center', fontdict=None, size=sz_text, color='k', bbox=dict(boxstyle='round', facecolor='w'))

    sns.despine(top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    fig.savefig(path_names[2] + 'final/MvsB_pmoth.pdf')
    plt.close(fig)
    print('Poisson Plot saved')
    
# if PLOT_MvsB:
#     # taus = [1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
#     data_name = '2018-02-16-aa'
#     # data_name = datasets[-1]
#     path_names = mf.get_directories(data_name=data_name)
#     print(data_name)
#     p = path_names[1]
#
#     plot_dprime = True
#
#     groups = np.load(p + 'VanRossum_groups_' + stim_type + '.npy').item()
#
#     p_moths = [[]] * len(taus)
#     for tt in range(len(taus)):
#         p_m = [[]] * len(duration)
#         p_b = [[]] * len(duration)
#         for k in range(len(duration)):
#             # ax = plt.subplot(len(duration), 1, k+1)
#             p_m[k] = groups[taus[tt]][k][0, :] / (groups[taus[tt]][k][1, :] + groups[taus[tt]][k][0, :])
#             # p_b[k] = groups[taus[tt]][k][1, :] / (groups[taus[tt]][k][1, :] + groups[taus[tt]][k][0, :])
#             # ax.imshow(ratio)
#         p_moths[tt] = p_m
#
#     # idx = [True, False, False, True, False, False, True, False, False, False, False, False, True]
#     tau_p = [1, 10, 50, 1000]  # taus used for plotting percentage
#     idx = []
#     for i in tau_p:
#         idx.append(taus.index(i))
#     p_moths = np.array(p_moths)[idx]
#
#     # d prime: bat = signal, moth = noise
#     d_prime = np.zeros(shape=(len(taus), len(duration)))
#     crit = np.zeros(shape=(len(taus), len(duration)))
#     area_d = np.zeros(shape=(len(taus), len(duration)))
#     beta = np.zeros(shape=(len(taus), len(duration)))
#
#     if stim_length == 'series':
#         idx_groups = 16
#     if stim_length == 'single':
#         idx_groups = 19
#
#     for i in range(len(taus)):
#         out = [[]] * len(duration)
#         for k in range(len(duration)):
#             a = groups[taus[i]][k]
#             cr = np.sum(a[0, :idx_groups])    # call=moth, matching=response=moth
#             miss = np.sum(a[0, idx_groups:])      # call=bat, matching=moth
#             fa = np.sum(a[1, :idx_groups])    # call=moth, matching=bat
#             hits = np.sum(a[1, idx_groups:])      # call=bat, matching=bat
#             out[k] = mf.dPrime(hits, miss, fa, cr)
#             d_prime[i, k] = out[k]['d']
#             crit[i, k] = out[k]['c']
#             area_d[i, k] = out[k]['Ad']
#             beta[i, k] = out[k]['beta']
#
#     mf.plot_settings()
#     # d prime plot
#     if plot_dprime:
#         # Create Grid
#         fig = plt.figure(figsize=(5.9, 2.9))
#         from mpl_toolkits.axes_grid1 import ImageGrid
#         grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
#                          nrows_ncols=(1, 2),
#                          label_mode='L',
#                          axes_pad=0.75,
#                          share_all=False,
#                          cbar_location="right",
#                          cbar_mode="each",
#                          cbar_size="3%",
#                          cbar_pad=0.05,
#                          aspect=False
#                          )
#         # Subplot caps
#         subfig_caps = 12
#         label_x_pos = 0.05
#         label_y_pos = 0.90
#         subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
#
#         # im1 = grid[0].imshow(d_prime, vmin=np.min(d_prime), vmax=3, interpolation='gaussian', cmap='jet', origin='lower')
#         # im2 = grid[1].imshow(crit, vmin=-1, vmax=1, interpolation='gaussian', cmap='seismic', origin='lower')
#         grid[0].text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=grid[0].transAxes, size=subfig_caps,
#                      color='black')
#         grid[1].text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=grid[1].transAxes, size=subfig_caps,
#                      color='black')
#
#         # Image Plot
#         x = duration
#         y = taus
#         X, Y = np.meshgrid(x, y)
#         # im1 = grid[0].pcolormesh(X, Y, d_prime, cmap='jet', vmin=np.min(d_prime)-0.5, vmax=3, shading='gouraud')
#         # im2 = grid[1].pcolormesh(X, Y, crit, cmap='seismic', vmin=-1, vmax=1, shading='gouraud')
#         im1 = grid[0].pcolormesh(X, Y, d_prime, cmap='jet', vmin=np.min(d_prime) - 0.5, vmax=2, shading='flat')
#         im2 = grid[1].pcolormesh(X, Y, crit, cmap='seismic', vmin=-1, vmax=1, shading='flat')
#
#         # grid[1].axvline(15, color='black', linestyle=':', linewidth=0.5)
#         # grid[0].axvline(15, color='black', linestyle=':', linewidth=0.5)
#         # grid[0].axhline(30, color='black', linestyle=':', linewidth=0.5)
#         # grid[1].axhline(30, color='black', linestyle=':', linewidth=0.5)
#
#         # grid[0].set_xscale('log')
#         grid[0].set_yscale('log')
#         # grid[1].set_xscale('log')
#         grid[1].set_yscale('log')
#
#         # Colorbar
#         cbar1 = grid[0].cax.colorbar(im1, ticks=np.arange(0, 2.1, 1))
#         cbar2 = grid[1].cax.colorbar(im2, ticks=[-1, -0.5, 0, 0.5, 1])
#         cbar1.ax.set_ylabel('d prime', rotation=270, labelpad=15)
#         cbar2.ax.set_ylabel('criterion', rotation=270, labelpad=10)
#         cbar1.solids.set_rasterized(True)  # Removes white lines
#         cbar2.solids.set_rasterized(True)  # Removes white lines
#
#         # Axes Labels
#         grid[0].set_ylabel('Tau [ms]')
#         fig.text(0.5, 0.075, 'Spike train duration [ms]', ha='center', fontdict=None)
#
#         # fig.set_size_inches(5.9, 1.9)
#         fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
#         figname = path_names[2] + 'dprime_MothsvsBats_' + stim_type + '_new.pdf'
#         fig.savefig(figname)
#         plt.close(fig)
#         print('d prime plot saved')
#
#     # Percentage Plot
#     # Create Grid
#     fig = plt.figure(figsize=(5.9, 3.9))
#     from mpl_toolkits.axes_grid1 import ImageGrid
#     grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
#                      nrows_ncols=(2, 2),
#                      label_mode='L',
#                      axes_pad=0.15,
#                      share_all=False,
#                      cbar_location="right",
#                      cbar_mode="single",
#                      cbar_size="3%",
#                      cbar_pad=0.15,
#                      aspect=False
#                      )
#     # Subplot caps
#     subfig_caps = 12
#     label_x_pos = 0.05
#     label_y_pos = 0.85
#     subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
#     i = 0
#     taus2 = np.array(taus)[idx]
#     if stim_length == 'series':
#         bats_region = 17
#     if stim_length == 'single':
#         bats_region = 20
#
#     for ax in grid:
#         y = duration
#         x = np.linspace(1, p_moths[i].shape[1], p_moths[i].shape[1])
#         X, Y = np.meshgrid(x, y)
#         im = ax.pcolormesh(X, Y, p_moths[i], cmap='jet', vmin=0, vmax=1, shading='flat', rasterized=True)
#
#         grid[i].axvline(bats_region, color='black', linestyle='-', linewidth=3)
#         grid[i].axvline(bats_region, color='white', linestyle='--', linewidth=1)
#
#         ax.set_xticks(np.arange(0, 30, 5))
#
#         grid[i].text(label_x_pos, label_y_pos, subfig_caps_labels[i], transform=grid[i].transAxes, size=subfig_caps,
#                      color='black')
#         grid[i].text(0.7, 0.05, r'$\tau$ = ' + str(taus2[i]) + ' ms', transform=grid[i].transAxes, size=6,
#                      color='black')
#         # grid[i].set_yscale('log')
#
#         i += 1
#
#     if stim_length == 'series':
#         grid[0].text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=grid[0].transAxes, size=subfig_caps,
#                      color='white')
#     # Colorbar
#     cbar = ax.cax.colorbar(im)
#     cbar.solids.set_rasterized(True)  # Removes white lines
#
#     # Axes Labels
#     fig.text(0.5, 0.05, 'Original call', ha='center', fontdict=None)
#     fig.text(0.025, 0.65, 'Spike train duration [ms]', ha='center', fontdict=None, rotation=90)
#     fig.text(0.965, 0.65, 'Percentage moth calls', ha='center', fontdict=None, rotation=270)
#
#     # fig.set_size_inches(5.9, 1.9)
#     fig.subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.9, wspace=0.1, hspace=0.1)
#     figname = path_names[2] + 'VanRossum_MothsvsBats_' + stim_type + '_new.pdf'
#     fig.savefig(figname)
#     plt.close(fig)

if PLOT_MVSB_DPRIME:
    boarder_series = 17
    boarder_single = 20
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    # Get Data
    data_vr_series_ext = np.load(path_names[1] + 'MvsB_VanRossum_' + 'series' + '_extended.npy').item()
    data_vr_single_ext = np.load(path_names[1] + 'MvsB_VanRossum_' + 'single' + '_extended.npy').item()
    data_d_series_ext = np.load(path_names[1] + 'MvsB_Distances_' + 'series' + '_extended.npy').item()
    data_d_single_ext = np.load(path_names[1] + 'MvsB_Distances_' + 'single' + '_extended.npy').item()
    data_vr_series = np.load(path_names[1] + 'MvsB_VanRossum_' + 'series' + '.npy').item()
    data_vr_single = np.load(path_names[1] + 'MvsB_VanRossum_' + 'single' + '.npy').item()
    data_d_series = np.load(path_names[1] + 'MvsB_Distances_' + 'series' + '.npy').item()
    data_d_single = np.load(path_names[1] + 'MvsB_Distances_' + 'single' + '.npy').item()

    # Plot percent moth
    mf.plot_settings()
    ax = [[]] * 18
    grid = matplotlib.gridspec.GridSpec(nrows=53, ncols=66)
    fig = plt.figure(figsize=(5.9, 4.9))
    ax[0] = plt.subplot(grid[0:10, 0:10])
    ax[1] = plt.subplot(grid[0:10, 14:24])

    ax[2] = plt.subplot(grid[0:10, 38:48])
    ax[3] = plt.subplot(grid[0:10, 52:62])

    ax[4] = plt.subplot(grid[14:24, 0:10])
    ax[5] = plt.subplot(grid[14:24, 14:24])

    ax[6] = plt.subplot(grid[14:24, 38:48])
    ax[7] = plt.subplot(grid[14:24, 52:62])

    ax[8] = plt.subplot(grid[28:38, 0:10])
    ax[9] = plt.subplot(grid[28:38, 14:24])

    ax[10] = plt.subplot(grid[28:38, 38:48])
    ax[11] = plt.subplot(grid[28:38, 52:62])

    ax[12] = plt.subplot(grid[42:52, 0:10])
    ax[13] = plt.subplot(grid[42:52, 14:24])

    ax[14] = plt.subplot(grid[42:52, 38:48])
    ax[15] = plt.subplot(grid[42:52, 52:62])

    # Colorbar
    ax[16] = plt.subplot(grid[0:10, 64:65])
    ax[17] = plt.subplot(grid[14:24, 64:65])


    # Image Grid
    y_series = taus
    x_series = np.arange(0, 2550, 50)
    x_series[0] = 10
    X_series, Y_series = np.meshgrid(x_series, y_series)

    y_single = taus
    x_single = np.arange(0, 255, 5)
    x_single[0] = 1
    X_single, Y_single = np.meshgrid(x_single, y_single)

    Xticks = [[]] * 4
    Xticks[0] = np.arange(0, 2550, 500)
    Xticks[1] = np.arange(0, 2550, 500)
    Xticks[2] = np.arange(0, 255, 50)
    Xticks[3] = np.arange(0, 255, 50)

    # Subplot caps
    subfig_caps = 12
    label_x_pos = -0.6
    label_y_pos = 1.05
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    color_map = 'jet'
    # Van Rossum d prime
    ax[0].pcolormesh(X_series, Y_series, data_vr_series['dprime'], cmap=color_map, vmin=0, vmax=2, shading='gouraud')
    ax[1].pcolormesh(X_series, Y_series, data_vr_series_ext['dprime'], cmap=color_map, vmin=0, vmax=2, shading='gouraud')
    ax[2].pcolormesh(X_single, Y_single, data_vr_single['dprime'], cmap=color_map, vmin=0, vmax=2, shading='gouraud')
    ax[3].pcolormesh(X_single, Y_single, data_vr_single_ext['dprime'], cmap=color_map, vmin=0, vmax=2, shading='gouraud')
    # Van Rossum criterion c
    c_color_map = 'seismic'
    ax[4].pcolormesh(X_series, Y_series, data_vr_series['c'], cmap=c_color_map, vmin=-1, vmax=1, shading='gouraud')
    ax[5].pcolormesh(X_series, Y_series, data_vr_series_ext['c'], cmap=c_color_map, vmin=-1, vmax=1, shading='gouraud')
    ax[6].pcolormesh(X_single, Y_single, data_vr_single['c'], cmap=c_color_map, vmin=-1, vmax=1, shading='gouraud')
    ax[7].pcolormesh(X_single, Y_single, data_vr_single_ext['c'], cmap=c_color_map, vmin=-1, vmax=1, shading='gouraud')

    # Log Axes
    import matplotlib.ticker
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)

    for i in range(8):
        ax[i].set_yscale('log')
        ax[i].yaxis.set_major_locator(locmaj)
        ax[i].yaxis.set_minor_locator(locmin)
        ax[i].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # ax[i].tick_params(axis='x', which='minor')
        # ax[i].set_yscale('symlog')

    # Distances d prime
    profiles = ['ISI', 'SYNC', 'DUR', 'COUNT']
    marks = ['o', 's', '', '']
    cc = ['orangered', 'teal', '0', '0']
    styles = ['-', '-', '--', ':']
    for i in range(4):
        ax[8].plot(x_series, data_d_series['dprime'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[9].plot(x_series, data_d_series_ext['dprime'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[10].plot(x_single, data_d_single['dprime'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[11].plot(x_single, data_d_single_ext['dprime'][i], marker='', color=cc[i], linestyle=styles[i], label=profiles[i])
        
    # Distances criterion c
    for i in range(4):
        ax[12].plot(x_series, data_d_series['c'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[13].plot(x_series, data_d_series_ext['c'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[14].plot(x_single, data_d_single['c'][i], marker='', color=cc[i], linestyle=styles[i])
        ax[15].plot(x_single, data_d_single_ext['c'][i], marker='', color=cc[i], linestyle=styles[i])

    # Colorbar
    norm1 = matplotlib.colors.Normalize(vmin=0, vmax=2)
    cb1 = matplotlib.colorbar.ColorbarBase(ax[16], cmap=color_map, norm=norm1)
    norm2 = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb2 = matplotlib.colorbar.ColorbarBase(ax[17], cmap=c_color_map, norm=norm2)

    cb1.set_ticks([0, 1, 2])
    cb2.set_ticks([-1, 0, 1])
    ax[16].text(6, 0.5, 'd prime', ha='center', va='center', fontdict=None, rotation=-90)
    ax[17].text(6, 0.5, 'Criterion c', ha='center', va='center', fontdict=None, rotation=-90)

    # cb1.set_ticklabels([0, 0.5, 1])

    # SERIES
    y_labels = [r'$\tau$ [ms]', r'$\tau$ [ms]', 'd prime', 'criterion c']
    la_y_pos = [20, 20, 1.5, 0.5]
    la_x_pos = [-1600, -1600, -1800, -1800]

    j = 0
    jj = 0
    for i in [0, 4, 8, 12]:  # series
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[j], transform=ax[i].transAxes, size=subfig_caps,
                   color='black')
        # ax[i].set_ylabel(y_labels[jj])
        ax[i].text(la_x_pos[jj], la_y_pos[jj], y_labels[jj], ha='center', va='center', fontdict=None, rotation=90)
        j += 2
        jj += 1
        for k in range(2):
            ax[i + k].set_xticks([0, 1000, 2000])
            ax[i + k].set_xticklabels([0, 1, 2])
        ax[i + 1].set_yticklabels([])

    # SINGLE
    la_y_pos = [20, 20, 1.5, 0.5]
    la_x_pos = [-160, -160, -180, -180]
    j = 1
    jj = 0
    for i in [2, 6, 10, 14]:  # single
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[j], transform=ax[i].transAxes, size=subfig_caps,
                   color='black')
        ax[i].text(la_x_pos[jj], la_y_pos[jj], y_labels[jj], ha='center', va='center', fontdict=None, rotation=90)
        j += 2
        jj += 1
        for k in range(2):
            ax[i + k].set_xticks([0, 100, 200])
            ax[i + k].set_xticklabels([0, 0.1, 0.2])
        ax[i + 1].set_yticklabels([])

    ax[8].set_ylim(0, 3)
    ax[9].set_ylim(0, 3)
    ax[10].set_ylim(0, 3)
    ax[11].set_ylim(0, 3)

    ax[8].set_yticks([0, 1, 2, 3])
    ax[9].set_yticks([0, 1, 2, 3])
    ax[10].set_yticks([0, 1, 2, 3])
    ax[11].set_yticks([0, 1, 2, 3])

    ax[12].set_ylim(-1, 2)
    ax[13].set_ylim(-1, 2)
    ax[14].set_ylim(-1, 2)
    ax[15].set_ylim(-1, 2)

    ax[12].set_yticks([-1, 0, 1, 2])
    ax[13].set_yticks([-1, 0, 1, 2])
    ax[14].set_yticks([-1, 0, 1, 2])
    ax[15].set_yticks([-1, 0, 1, 2])

    # ax[18].legend()
    ax[11].legend(frameon=False ,bbox_to_anchor=(1.1, 1.05))

    sz_text = 12
    ax[0].text(1200, 10000, 'Call series', ha='center', va='center', fontdict=None, size=sz_text)
    ax[1].text(1200, 10000, 'Cs extended', ha='center', va='center', fontdict=None, size=sz_text)
    ax[2].text(120, 10000, 'Single calls', ha='center', va='center', fontdict=None, size=sz_text)
    ax[3].text(120, 10000, 'Sc extended', ha='center', va='center', fontdict=None, size=sz_text)

    for i in range(8, 16):
        sns.despine(ax=ax[i])

    # fig.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(path_names[2] + 'final/MvsB_dprime.pdf')
    plt.close(fig)
    print('Poisson d prime Plot saved')

if PLOT_VR:
    # data_name = '2018-02-09-aa'
    data_name = '2018-02-16-aa'
    # data_name = datasets[-1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]

    matches = np.load(p + 'VanRossum_matches_' + stim_type + '.npy').item()

    # Plot
    mf.plot_settings()
    from mpl_toolkits.axes_grid1 import ImageGrid
    # Set up figure and image grid
    fig = plt.figure(figsize=(5.9, 3.9))

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 3),
                     label_mode='L',
                     axes_pad=0.15,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    # Find taus and duration
    if stim_length == 'series':
        taus_selected = [1, 10, 1000, 10, 10, 10]
        dur_selected = [1000, 1000, 1000, 50, 500, 2000]
    if stim_length == 'single':
        taus_selected = [1, 10, 100, 10, 10, 10]
        dur_selected = [200, 200, 200, 5, 50, 250]
    idx_dur = np.where(np.isin(duration, dur_selected))[0]
    idx_dur = [idx_dur[2], idx_dur[2], idx_dur[2], idx_dur[0], idx_dur[1], idx_dur[3]]

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.85
    label_y_pos = 0.85
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    k = 0
    for ax in grid:
        im = ax.imshow(matches[taus_selected[k]][idx_dur[k]], vmin=0, vmax=20, cmap='Greys')
        grid[k].text(label_x_pos, label_y_pos, subfig_caps_labels[k], transform=grid[k].transAxes, size=subfig_caps, color='black')
        grid[k].text(0.1, 0.1, r'$\tau$ = '+str(taus_selected[k])+' ms', transform=grid[k].transAxes, size=6,
                     color='black')
        grid[k].text(0.1, 0.05, 'dur = ' + str(dur_selected[k]) + ' ms', transform=grid[k].transAxes, size=6,
                     color='black')
        ax.set_xticks(np.arange(0, 20, 5))
        ax.set_yticks(np.arange(0, 20, 5))
        k += 1

    # Colorbar
    cbar = ax.cax.colorbar(im)
    # cbar.ax.set_ylabel('Spike trains', rotation=270)
    cbar.solids.set_rasterized(True)  # Removes white lines

    # Axes Labels
    fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
    fig.text(0.05, 0.55, 'Matched call', ha='center', fontdict=None, rotation=90)
    fig.text(0.96, 0.55, 'Spike trains', ha='center', fontdict=None, rotation=270)
    # Save Plot to HDD
    # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.4, hspace=0.4)
    figname = path_names[2] + 'VanRossum_Matrix_' + stim_type + '.pdf'
    fig.savefig(figname)
    plt.close(fig)
    print('VanRossum Matrix Plot saved')

if PLOT_DISTANCES:
    data_name = '2018-02-16-aa'
    # data_name = datasets[-1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    matches = np.load(p + 'distances_matches_' + stim_type + '.npy').item()
    prof = 'ISI'
    matches = matches[prof]

    # Plot
    mf.plot_settings()
    from mpl_toolkits.axes_grid1 import ImageGrid
    # Set up figure and image grid
    fig = plt.figure(figsize=(5.9, 3.9))

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 3),
                     label_mode='L',
                     axes_pad=0.15,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    all_axes = []
    dur_p = [10, 50, 100, 500, 1000, 2000]  # taus used for plotting percentage
    idx = []
    for i in dur_p:
        idx.append(duration.index(i))

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.85
    label_y_pos = 0.85
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    k = 0
    for ax in grid:
        im = ax.imshow(matches[idx[k]], vmin=0, vmax=20, cmap='Greys')
        all_axes.append(ax)
        grid[k].text(label_x_pos, label_y_pos, subfig_caps_labels[k], transform=grid[k].transAxes, size=subfig_caps, color='black')
        # grid[k].text(0.1, 0.1, r'$\tau$ = '+str(taus[taus_idx[k]])+' ms', transform=grid[k].transAxes, size=6,
        #              color='black')
        # grid[k].text(0.1, 0.05, 'dur = ' + str(duration[dur_idx[k]]) + ' ms', transform=grid[k].transAxes, size=6,
        #              color='black')

        k += 1


    # Colorbar
    cbar = ax.cax.colorbar(im)
    # cbar.ax.set_ylabel('Spike trains', rotation=270)
    cbar.solids.set_rasterized(True)  # Removes white lines

    # Axes Labels
    fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
    fig.text(0.05, 0.55, 'Matched call', ha='center', fontdict=None, rotation=90)
    fig.text(0.96, 0.55, 'Spike trains', ha='center', fontdict=None, rotation=270)
    # Save Plot to HDD
    # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.4, hspace=0.4)
    figname = path_names[2] + '/Distances_Matrix_' + stim_type + '.pdf'
    fig.savefig(figname)
    plt.close(fig)
    print('Distances Matrix Plot saved')

if PLOT_CORRECT:
    data_name = '2018-02-16-aa'
    # data_name = datasets[-1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    if extended:
        figname = path_names[2] + 'Distances_Correct_new_' + stim_type + '_extended.pdf'
        matches = np.load(p + 'distances_correct_' + stim_type + '_extended.npy')
        rand_machtes = np.load(p + 'distances_rand_correct_' + stim_type + '_extended.npy')
    else:
        figname = path_names[2] + 'Distances_Correct_new_' + stim_type + '.pdf'
        matches = np.load(p + 'distances_correct_' + stim_type + '.npy')
        rand_machtes = np.load(p + 'distances_rand_correct_' + stim_type + '.npy')
    # Plot
    mf.plot_settings()
    # fig = plt.figure(figsize=(5.9, 3.9))
    fig, ax = plt.subplots()

    marks = ['', 'o', 'v', 's', '']
    cc = ['0', 'orangered', 'navy', 'teal', '0']
    styles = [':', '-', '-', '-', '--']
    for k in range(len(profs)):
        ax.plot(duration, matches[:, k], marker=marks[k], label=profs[k], color=cc[k], linestyle=styles[k], markersize=3)

    ax.plot(duration, rand_machtes[:, 1], 'k', linewidth=2, label='Random')
    rand_mean = np.round(np.mean(rand_machtes[:, 1]), 2)
    # ax.text(2000, 0.1, 'random mean = ' + str(rand_mean), size=6, color='black')

    ax.set_xlabel('Spike train duration [ms]')
    ax.set_ylabel('Correct')
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(0, 1)
    if stim_length == 'series':
        ax.text(2000, 0.1, 'random mean = ' + str(rand_mean), size=6, color='black')
        ax.set_xticks(np.arange(0, duration[-1]+100, 500))
        ax.set_xlim(-0.2, duration[-1]+100)
    if stim_length == 'single':
        ax.text(200, 0.07, 'random mean = ' + str(rand_mean), size=6, color='black')
        ax.set_xticks(np.arange(0, duration[-1] + 10, 50))
        ax.set_xlim(-0.2, duration[-1] + 10)
    sns.despine()
    ax.legend(frameon=False)
    # Save Plot to HDD
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.4, hspace=0.4)
    fig.set_size_inches(5.9, 2.9)
    fig.savefig(figname)
    plt.close(fig)
    print('Distances Matrix Plot saved')

if PLOT_VR_TAUVSDUR:
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    # diff_extended = True
    # if extended:
    #     vr_series = np.load(p + 'VanRossum_correct_' + 'moth_series_selected' + '_extended.npy')
    #     vr_series_dur = np.load(p + 'VanRossum_correct_' + 'moth_series_selected' + '.npy')
    #     vr_single = np.load(p + 'VanRossum_correct_' + 'moth_single_selected' + '_extended.npy')
    #     vr_single_dur = np.load(p + 'VanRossum_correct_' + 'moth_single_selected' + '.npy')
    #     figname = path_names[2] + 'VanRossum_TauVSDur_extended.pdf'
    #     if diff_extended:
    #         vr_series = vr_series_dur - vr_series
    #         vr_single = vr_single_dur - vr_single
    #         figname = path_names[2] + 'VanRossum_TauVSDur_extended_diff.pdf'
    # elif stim_type == 'poisson':
    #     vr_series = np.load(p + 'VanRossum_correct_' + 'poisson' + '.npy')
    #     vr_single = np.load(p + 'VanRossum_correct_' + 'poisson' + '.npy')
    #     figname = path_names[2] + 'VanRossum_TauVSDur_poisson.pdf'
    # else:
    #     vr_series = np.load(p + 'VanRossum_correct_' + 'moth_series_selected' + '.npy')
    #     vr_single = np.load(p + 'VanRossum_correct_' + 'moth_single_selected' + '.npy')
    #     figname = path_names[2] + 'VanRossum_TauVSDur.pdf'

    # Get all data
    s_types = ['moth_series_selected', 'moth_series_selected_extended', 'moth_single_selected',
               'moth_single_selected_extended']
    data = {}
    for k in range(len(s_types)):
        data.update({s_types[k]: np.load(p + 'VanRossum_correct_' + s_types[k] + '.npy')})

    # Plot
    mf.plot_settings()
    ax = [[]] * 5
    grid = matplotlib.gridspec.GridSpec(nrows=24, ncols=26)
    fig = plt.figure(figsize=(5.9, 4.9))
    ax[0] = plt.subplot(grid[0:10, 0:10])
    ax[1] = plt.subplot(grid[0:10, 13:23])
    ax[2] = plt.subplot(grid[13:23, 0:10])
    ax[3] = plt.subplot(grid[13:23, 13:23])
    # Colorbar ax
    ax[4] = plt.subplot(grid[0:23, 24:25])

    # Image Grid
    XX = [[]] * 4
    x_single = list(np.arange(0, 255, 5))
    x_single[0] = 1
    x_series = list(np.arange(0, 2550, 50))
    x_series[0] = 10
    XX[0] = x_series
    XX[1] = x_series
    XX[2] = x_single
    XX[3] = x_single
    y = taus

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.025
    label_y_pos = 1.05
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    grid_color = '0.75'
    grid_linewidth = 0.5

    Xticks = [[]] * 4
    Xticks[0] = np.arange(0, 2550, 500)
    Xticks[1] = np.arange(0, 2550, 500)
    Xticks[2] = np.arange(0, 255, 50)
    Xticks[3] = np.arange(0, 255, 50)

    # ColorMeshPlot
    for i in range(len(s_types)):
        X, Y = np.meshgrid(XX[i], y)
        # im = ax[i].pcolormesh(X, Y, data[s_types[i]].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
        im = ax[i].pcolormesh(X, Y, data[s_types[i]].T, cmap='jet', vmin=0, vmax=1, shading='flat', rasterized=True)

        ax[i].set_yscale('log')
        ax[i].set_xticks(Xticks[i])
        # Subplot caps
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[i], transform=ax[i].transAxes, size=subfig_caps,
                 color='black')
        # ax[i].grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax[4], cmap='jet', norm=norm)
    # cb1.set_label('Correct')

    fig.text(0.5, 0.025, 'Spike train duration [ms]', ha='center', fontdict=None)
    fig.text(0.025, 0.55, r'$\tau$ [ms]', ha='center', fontdict=None, rotation=90)
    fig.text(0.935, 0.55, 'Correct', ha='center', fontdict=None, rotation=-90)

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(path_names[2] + 'final/VanRossum_TauVsDur.pdf')
    plt.close(fig)
    print('TauvsDur Plot saved')
    exit()
    # OLD ==============================================================================================================
    mf.plot_settings()
    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=43)
    fig = plt.figure(figsize=(5.9, 2.9))
    ax1 = plt.subplot(grid[0:19])
    ax2 = plt.subplot(grid[21:40])
    ax3 = plt.subplot(grid[41])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    # Image Plot
    x_single = list(np.arange(0, 255, 5))
    x_single[0] = 1
    x_series = list(np.arange(0, 2550, 50))
    x_series[0] = 10
    y = taus
    X_single, Y_single = np.meshgrid(x_single, y)
    X_series, Y_series = np.meshgrid(x_series, y)

    im1 = ax1.pcolormesh(X_series, Y_series, vr_series.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
    im2 = ax2.pcolormesh(X_single, Y_single, vr_single.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
    # im1 = ax1.pcolormesh(X_series, Y_series, vr_series.T, cmap='jet', vmin=0, vmax=1, shading='flat')
    # im2 = ax2.pcolormesh(X_single, Y_single, vr_single.T, cmap='jet', vmin=0, vmax=1, shading='flat')

    # grid[0].axhline(200, color='black', linestyle=':', linewidth=0.5)

    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Axes Limits
    ax2.set_yticks([])

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax3, cmap='jet', norm=norm)
    cb1.set_label('Correct')
    # cbar2 = plt.colorbar(im2, ticks=np.arange(0, 1.1, 0.2))
    # cbar2.ax.set_ylabel('Correct', rotation=270, labelpad=10)
    # cbar2.solids.set_rasterized(True)  # Removes white lines

    # Axes Labels
    ax1.set_ylabel('Tau [ms]')
    fig.text(0.5, 0.075, 'Spike train duration [ms]', ha='center', fontdict=None)

    # Subfig Caps
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                 color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                 color='black')

    # fig.set_size_inches(5.9, 1.9)
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(figname)
    plt.close(fig)

if ISI:
    data_name = '2018-02-16-aa'
    # data_name = datasets[-1]
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    print(stim_type)
    print('extended: ' + str(extended))
    method = 'exp'
    path_save = path_names[1]
    # plot_correct = False
    save_fig = False
    dist_profs = {}
    matches = {}

    correct = np.zeros((len(duration), len(profs)))
    rand_correct = np.zeros((len(duration), len(profs)))
    for p in tqdm(range(len(profs)), desc='Profiles'):
        distances_all = [[]] * len(duration)
        mm = [[]] * len(duration)
        # Parallel loop through all durations
        r = Parallel(n_jobs=3)(delayed(mf.isi_matrix)(path_names, duration[i]/1000, boot_sample=nsamples,
                                                      stim_type=stim_type, profile=profs[p], save_fig=save_fig,
                                                      extended=extended) for i in range(len(duration)))

        # mm_mean, correct_matches, distances_per_boot, rand_correct_matches = mf.isi_matrix(path_names, duration[5]/1000, boot_sample=nsamples,stim_type=stim_type, profile=profs[p], save_fig=save_fig)

        # Put values from parallel loop into correct variables
        for q in range(len(duration)):
            mm[q] = r[q][0]
            correct[q, p] = r[q][1]
            rand_correct[q, p] = r[q][3]
            distances_all[q] = r[q][2]
        dist_profs.update({profs[p]: distances_all})
        matches.update({profs[p]: mm})

    # Save to HDD
    if extended:
        np.save(path_save + 'distances_' + stim_type + '_extended.npy', dist_profs)
        np.save(path_save + 'distances_correct_' + stim_type + '_extended.npy', correct)
        np.save(path_save + 'distances_rand_correct_' + stim_type + '_extended.npy', rand_correct)
        np.save(path_save + 'distances_matches_' + stim_type + '_extended.npy', matches)
    else:
        np.save(path_save + 'distances_' + stim_type + '.npy', dist_profs)
        np.save(path_save + 'distances_correct_' + stim_type + '.npy', correct)
        np.save(path_save + 'distances_rand_correct_' + stim_type + '.npy', rand_correct)
        np.save(path_save + 'distances_matches_' + stim_type + '.npy', matches)

    #
    # if plot_correct:
    #     for k in range(len(profs)):
    #         plt.subplot(np.ceil(len(profs)/2), 2, k+1)
    #         plt.plot(duration, correct[:, k], 'ko-')
    #         plt.xlabel('Spike Train Length [ms]')
    #         plt.ylabel('Correct [' + profs[k] + ']')
    #     plt.show()

if PULSE_TRAIN_ISI:
    save_plot = True
    p = path_names[1]
    dists = np.load(p + 'distances.npy').item()

    if stim_type == 'moth_single':
        stim_type2 = 'naturalmothcalls'
    elif stim_type == 'moth_series':
        stim_type2 = 'callseries/moths'
    else:
        print('No Pulse Trains available for: ' + stim_type)
        exit()

    fs = 480*1000
    for j in tqdm(range(len(profs)), desc='Profiles', leave=False):
        distances_all = dists[profs[j]]
        for i in tqdm(range(len(duration)), desc='PulseTrainDistance'):
            distances = distances_all[i]
            duration[i] = duration[i] / 1000
            calls, calls_names = mf.mattopy(stim_type2, fs)
            d_pulses_isi = mf.pulse_train_matrix(calls, duration[i], profs[j])

            d_st_isi = distances[len(distances)-1][0]
            for k in range(len(distances)-1):
                d_st_isi = d_st_isi + distances[k][0]
            d_st_isi = d_st_isi / len(distances)

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(d_pulses_isi)
            no_lims = True
            if profs[j] == 'COUNT':
                no_lims = False
            if profs[j] == 'DUR':
                no_lims = False
            if no_lims:
                plt.clim(0, 1)
            plt.xticks(np.arange(0, len(d_pulses_isi), 1))
            plt.yticks(np.arange(0, len(d_pulses_isi), 1))
            plt.xlabel('Original Call')
            plt.ylabel('Matched Call')
            plt.colorbar(fraction=0.04, pad=0.02)
            plt.title(profs[j] + ': Pulse Trains [' + str(duration[i]*1000) + ' ms]')

            plt.subplot(1, 2, 2)
            plt.imshow(d_st_isi)
            if no_lims:
                plt.clim(0, 1)
            plt.xticks(np.arange(0, len(d_st_isi), 1))
            plt.yticks(np.arange(0, len(d_st_isi), 1))
            plt.xlabel('Original Call')
            plt.ylabel('Matched Call')
            plt.colorbar(fraction=0.04, pad=0.02)
            plt.title(profs[j] + ': Spike Trains [' + str(duration[i]*1000) + ' ms] (boot = ' + str(nsamples) + ')')
            # plt.tight_layout()

            if save_plot:
                # Save Plot to HDD
                figname = p + 'pulseVSspike_train_' + profs[j] + '_' + str(duration[i]*1000) + 'ms_.png'
                fig = plt.gcf()
                fig.set_size_inches(20, 10)
                fig.savefig(figname, bbox_inches='tight', dpi=300)
                plt.close(fig)
                # print('Plot saved')
            else:
                plt.show()

            '''
            # ISI Threshold?
            for i in range(len(d_st_isi)):
                self_dist = d_st_isi[i, i]
                the_rest = d_st_isi[i, :]
                the_rest = np.delete(the_rest, i)
                mean_rest = np.mean(the_rest)
                std_rest = np.std(the_rest)
                diff_mean_rest = abs(mean_rest - self_dist)
                diff_min_rest = abs(np.min(the_rest) - self_dist)
                per_rest = np.percentile(the_rest, 50)
                count = len(np.where(the_rest < self_dist+0.1)[0])
                # print(str(i) + ': ' + str(diff_mean_rest) + ' | ' + str(count))
                # print(str(i) + ': ' + str(self_dist) + ' | ' + str(per_rest))
                # print(str(i) + ': ' + str(self_dist) + ' | ' + str(np.min(the_rest)))
                print(str(i) + ': ' + str(diff_min_rest) + ' | ' + str(count))
            '''

# if PULSE_TRAIN_VANROSSUM:
#     # Try to load e pulses from HDD
#     data_name = '2018-02-09-aa'
#     path_names = mf.get_directories(data_name=data_name)
#     print(data_name)
#     method = 'exp'
#     p = path_names[1]
#
#     vanrossum = np.load(p + 'VanRossum_' + stim_type + '.npy').item()
#     spike_distances = np.load(p + 'distances_' + stim_type + '.npy').item()
#
#     # tau = taus[9]  # select tau value
#
#     # Convert matlab files to pyhton
#     fs = 480 * 1000  # sampling of audio recordings
#     calls, calls_names = mf.mattopy(stim_type, fs)
#
#     # Convert Pulse Trains to E Pulses
#     taus_pulses = [1, 10, 530, 10, 10, 10]
#     taus_pulses = [1, 10, 200, 10, 10, 10]
#     duration_pulses = [1000, 1000, 1000, 50, 500, 2000]
#     duration_pulses = [100, 100, 100, 5, 50, 200]
#     duration_in_samples = (np.array(duration_pulses) / 1000) / dt
#     e_pulses = [[]] * len(taus_pulses)
#     for t in range(len(taus_pulses)):
#         e_pulses[t] = mf.pulse_trains_to_e_pulses(calls, taus_pulses[t] / 1000, dt)
#
#     # Compute Van Rossum Matrix for Pulse Trains
#     pulses_vr = [[]] * len(taus_pulses)
#     for t in range(len(taus_pulses)):
#         p_vr = np.zeros((len(e_pulses[t]), len(e_pulses[t])))
#         for k in range(len(e_pulses[t])):
#             for i in range(len(e_pulses[t])):
#                 p_vr[k, i] = mf.vanrossum_distance(e_pulses[t][k][0:int(duration_in_samples[t])],
#                                                    e_pulses[t][i][0:int(duration_in_samples[t])], dt,
#                                                    taus_pulses[t] / 1000)
#         pulses_vr[t] = p_vr / np.max(p_vr)  # normalized
#
#     # Compute Other Distance Metrics for Pulse Trains
#     profiles = ['ISI', 'SYNC', 'DUR', 'COUNT']
#     pulses_distances = [[]] * len(profiles)
#     for prof in range(len(profiles)):
#         p_distances = [[]] * len(duration)
#         for j in range(len(duration)):
#             p_distances[j] = mf.pulse_train_matrix(calls, duration[j] / 1000, profiles[prof])
#         pulses_distances[prof] = p_distances
#
#     # Compute Mean (over boots) of VanRossum for Spike Trains
#     idx = []
#     for i in duration_pulses:
#         idx.append(duration.index(i))
#     spikes_vr = [[]] * len(taus_pulses)
#     for t in range(len(taus_pulses)):
#         distances = vanrossum[taus_pulses[t]][idx[t]]
#         sp_vr = distances[len(distances) - 1][0]
#         for k in range(len(distances) - 1):
#             sp_vr = sp_vr + distances[k][0]
#         sp_vr = sp_vr / len(distances)
#         # Norm
#         sp_vr = sp_vr / np.max(sp_vr)
#         spikes_vr[t] = sp_vr
#
#     # Compute Mean (over boots) of  other Distances for Spike Trains
#     if stim_length == 'single':
#         selected_duration = 150
#     if stim_length == 'series':
#         selected_duration = 1500
#
#     a = np.where(np.array(duration) == selected_duration)
#     dur_d = a[0][0]
#     print(duration[dur_d])
#     spikes_d = [[]] * len(profiles)
#     for prof in range(len(profiles)):
#         distances = spike_distances[profiles[prof]][dur_d]
#         sp_d = distances[len(distances) - 1][0]
#         for k in range(len(distances) - 1):
#             sp_d = sp_d + distances[k][0]
#         sp_d = sp_d / len(distances)
#         spikes_d[prof] = sp_d
#
#     # Plot
#     mf.plot_settings()
#     from mpl_toolkits.axes_grid1 import ImageGrid
#     plot_name = ['/VanRossum_SpikeTrains_', '/VanRossum_PulseTrains_', '/Distances_SpikeTrains_', '/Distances_PulseTrains_']
#     plot_data = [spikes_vr, pulses_vr, spikes_d, pulses_distances]
#     plot_size = [(2, 3), (2, 3), (2, 2), (2, 2)]
#     method = ['vr', 'vr', 'sd', 'pd']
#     cbar_mode = ['single', 'single', 'each', 'each']
#     cbar_labels = ['ISI distance', 'SYNC value', 'Difference [s]', 'Difference [count]']
#     figure_sizes = [5.9, 5.9, 3.9, 3.9]
#     # axes_labels = []
#     for p in range(4):
#         # Set up figure and image grid
#         fig = plt.figure(figsize=(figure_sizes[p], 3.9))
#
#         grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
#                          nrows_ncols=plot_size[p],
#                          label_mode='L',
#                          axes_pad=0.75,
#                          share_all=True,
#                          cbar_location="right",
#                          cbar_mode=cbar_mode[p],
#                          cbar_size="3%",
#                          cbar_pad=0.15,
#                          )
#         # Create Grid
#         if method is 'vr':
#             grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
#
#         # Add data to image grid
#         all_axes = []
#
#         # Subplot caps
#         subfig_caps = 12
#         label_x_pos = 0.85
#         label_y_pos = 0.85
#         subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#         k = 0
#         for ax in grid:
#             if method[p] is 'vr':
#                 im = ax.imshow(plot_data[p][k], vmin=0, vmax=np.max(plot_data[p][k]), cmap='viridis')
#                 grid[k].text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[k]) + ' ms', transform=grid[k].transAxes, size=6,
#                              color='white')
#                 grid[k].text(0.1, 0.05, 'dur = ' + str(duration_pulses[k]) + ' ms', transform=grid[k].transAxes, size=6,
#                              color='white')
#
#             elif method[p] is 'pd':
#                 pd_limit = [1, 1, np.max(plot_data[p][k][dur_d]), np.max(plot_data[p][k][dur_d])]
#                 im = ax.imshow(plot_data[p][k][dur_d], vmin=0, vmax=pd_limit[k],  cmap='viridis')
#                 cbar = ax.cax.colorbar(im)
#                 cbar.ax.set_ylabel(cbar_labels[k], rotation=270, labelpad=15)
#                 cbar.solids.set_rasterized(True)  # Removes white lines
#                 grid[k].text(0.1, 0.05, 'dur = ' + str(selected_duration) + ' ms', transform=grid[k].transAxes, size=6,
#                              color='white')
#             else:
#                 pd_limit = [1, 1, np.max(plot_data[p][k]), np.max(plot_data[p][k])]
#                 im = ax.imshow(plot_data[p][k], vmin=0, vmax=pd_limit[k], cmap='viridis')
#                 cbar = ax.cax.colorbar(im)
#                 cbar.ax.set_ylabel(cbar_labels[k], rotation=270, labelpad=15)
#                 cbar.solids.set_rasterized(True)  # Removes white lines
#                 grid[k].text(0.1, 0.05, 'dur = ' + str(selected_duration) + ' ms', transform=grid[k].transAxes, size=6,
#                              color='white')
#
#             all_axes.append(ax)
#             grid[k].set_xticks(np.arange(0, 20, 5))
#             grid[k].set_yticks(np.arange(0, 20, 5))
#
#             grid[k].text(label_x_pos, label_y_pos, subfig_caps_labels[k], transform=grid[k].transAxes, size=subfig_caps,
#                          color='white')
#             k += 1
#
#         # Colorbar
#         if method[p] is 'vr':
#             cbar = ax.cax.colorbar(im)
#             cbar.ax.set_ylabel('Normalized Van Rossum Distance', rotation=270)
#             cbar.solids.set_rasterized(True)  # Removes white lines
#
#         # Axes Labels
#         fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
#         fig.text(0.05, 0.55, 'Matched call', ha='center', fontdict=None, rotation=90)
#         # Save Plot to HDD
#         # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.7, wspace=0.4, hspace=0.4)
#         figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + plot_name[p] + stim_type + '.pdf'
#         fig.savefig(figname)
#         plt.close(fig)
#
#     exit()
#     fig = plt.figure()
#     ax1 = plt.subplot(1, 2, 1)
#     ax2 = plt.subplot(1, 2, 2)
#     im1 = ax1.imshow(pulses_vr)
#     im2 = ax2.imshow(spikes_vr)
#     fig.colorbar(im1, ax=ax1)
#     fig.colorbar(im2, ax=ax2)
#     plt.show()
#     embed()
#     exit()

if PULSE_TRAIN_VANROSSUM:
    # Try to load e pulses from HDD
    # data_name = '2018-02-09-aa'
    # data_name = '2018-02-20-aa'
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    method = 'exp'
    p = path_names[1]

    vanrossum = np.load(p + 'VanRossum_' + stim_type + '.npy').item()
    spike_distances = np.load(p + 'distances_' + stim_type + '.npy').item()

    # tau = taus[9]  # select tau value

    # Convert matlab files to pyhton
    fs = 480 * 1000  # sampling of audio recordings
    calls, calls_names = mf.mattopy(stim_type, fs)

    # Convert Pulse Trains to E Pulses
    if stim_length is 'series':
        taus_pulses = [1, 10, 530, 10, 10, 10]
        duration_pulses = [1000, 1000, 1000, 50, 500, 2000]
    if stim_length is 'single':
        taus_pulses = [1, 10, 200, 10, 10, 10]
        duration_pulses = [100, 100, 100, 5, 50, 200]
    duration_in_samples = (np.array(duration_pulses) / 1000) / dt
    e_pulses = [[]] * len(taus_pulses)
    for t in range(len(taus_pulses)):
        e_pulses[t] = mf.pulse_trains_to_e_pulses(calls, taus_pulses[t] / 1000, dt)

    # Compute Van Rossum Matrix for Pulse Trains
    pulses_vr = [[]] * len(taus_pulses)
    for t in range(len(taus_pulses)):
        p_vr = np.zeros((len(e_pulses[t]), len(e_pulses[t])))
        for k in range(len(e_pulses[t])):
            for i in range(len(e_pulses[t])):
                p_vr[k, i] = mf.vanrossum_distance(e_pulses[t][k][0:int(duration_in_samples[t])],
                                                   e_pulses[t][i][0:int(duration_in_samples[t])], dt,
                                                   taus_pulses[t] / 1000)
        pulses_vr[t] = p_vr / np.max(p_vr)  # normalized

    # Compute Other Distance Metrics for Pulse Trains
    profiles = ['ISI', 'SYNC', 'DUR', 'COUNT']
    pulses_distances = [[]] * len(profiles)
    for prof in range(len(profiles)):
        p_distances = [[]] * len(duration)
        for j in range(len(duration)):
            p_distances[j] = mf.pulse_train_matrix(calls, duration[j] / 1000, profiles[prof])
        pulses_distances[prof] = p_distances

    # Compute Mean (over boots) of VanRossum for Spike Trains
    idx = []
    for i in duration_pulses:
        idx.append(duration.index(i))
    spikes_vr = [[]] * len(taus_pulses)
    for t in range(len(taus_pulses)):
        distances = vanrossum[taus_pulses[t]][idx[t]]
        sp_vr = distances[len(distances) - 1][0]
        for k in range(len(distances) - 1):
            sp_vr = sp_vr + distances[k][0]
        sp_vr = sp_vr / len(distances)
        # Norm
        sp_vr = sp_vr / np.max(sp_vr)
        spikes_vr[t] = sp_vr

    dursCS = [100, 200, 500, 1000, 1500, 2000]
    dursSC = [10, 20, 50, 100, 150, 200]

    for dd in range(len(dursCS)):
        # Compute Mean (over boots) of  other Distances for Spike Trains
        if stim_length == 'single':
            selected_duration = dursSC[dd]
        if stim_length == 'series':
            selected_duration = dursCS[dd]

        a = np.where(np.array(duration) == selected_duration)
        dur_d = a[0][0]
        print('duration is ' + str(duration[dur_d]) + ' ms')
        spikes_d = [[]] * len(profiles)
        for prof in range(len(profiles)):
            distances = spike_distances[profiles[prof]][dur_d]
            sp_d = distances[len(distances) - 1][0]
            for k in range(len(distances) - 1):
                sp_d = sp_d + distances[k][0]
            sp_d = sp_d / len(distances)
            spikes_d[prof] = sp_d

        # Plot
        mf.plot_settings()
        plot_name = ['/VanRossum_SpikeTrains_', '/VanRossum_PulseTrains_', '/Distances_SpikeTrains_', '/Distances_PulseTrains_']
        plot_data = [spikes_vr, pulses_vr, spikes_d, pulses_distances]
        plot_size = [(2, 3), (2, 3), (2, 2), (2, 2)]
        method = ['vr', 'vr', 'sd', 'pd']
        cbar_mode = ['single', 'single', 'each', 'each']
        cbar_labels = ['ISI', 'SYNC', 'DUR [s]', 'Norm. COUNT']
        figure_sizes = [5.9, 5.9, 3.9, 3.9]
        cc = 'white'

        fig = plt.figure(figsize=(5.9, 2.9))
        grid_step = 12
        x1 = 0
        x2 = 9
        y1 = 4
        y2 = 14
        grid = matplotlib.gridspec.GridSpec(nrows=y2+grid_step+1, ncols=x2+3*grid_step+1)

        ax1 = plt.subplot(grid[y1:y2, x1:x2])
        cb1 = plt.subplot(grid[2, x1:x2])

        ax2 = plt.subplot(grid[y1:y2, x1+grid_step:x2+grid_step])
        cb2 = plt.subplot(grid[2, x1+grid_step:x2+grid_step])

        ax3 = plt.subplot(grid[y1:y2, x1 + 2*grid_step:x2 + 2*grid_step])
        cb3 = plt.subplot(grid[2, x1 + 2*grid_step:x2 + 2*grid_step])

        ax4 = plt.subplot(grid[y1:y2, x1 + 3 * grid_step:x2 + 3 * grid_step])
        cb4 = plt.subplot(grid[2, x1 + 3 * grid_step:x2 + 3 * grid_step])

        ax5 = plt.subplot(grid[y1+grid_step:y2+grid_step, x1:x2])
        ax6 = plt.subplot(grid[y1+grid_step:y2+grid_step, x1+grid_step:x2+grid_step])
        ax7 = plt.subplot(grid[y1+grid_step:y2+grid_step, x1+2*grid_step:x2+2*grid_step])
        ax8 = plt.subplot(grid[y1+grid_step:y2+grid_step, x1+3*grid_step:x2+3*grid_step])

        # cb8 = plt.subplot(grid[52, 3])

        # Color Range Settings
        # DUR metric
        if stim_length is 'series':
            c3_settings = [0, 0.5, 1]
            if dursCS[dd] == 200:
                c3_settings = [0, 0.1, 0.2]

        if stim_length is 'single':
            c3_settings = [0, 0.05, 0.1]
            if dursSC[dd] == 20:
                c3_settings = [0, 0.01, 0.02]

        # COUNT metric
        if stim_length is 'series':
            c4_settings = [0, .5, 1]
            if dursCS[dd] == 200:
                c4_settings = [0, .5, 1]
        if stim_length is 'single':
            c4_settings = [0, .5, 1]

        # Normalize
        COUNT_S = plot_data[2][3] / np.max(plot_data[2][3])
        COUNT_P = plot_data[3][3][dur_d] / np.max(plot_data[3][3][dur_d])

        # if dursCS[dd] != 1000:
        #     continue
        #
        # if dursCS[dd] == 1000:
        #     plt.close('all')
        #
        #     ISI_diff = abs(plot_data[3][0][dur_d] - plot_data[2][0])
        #     ISI_ratio = plot_data[3][0][dur_d] / plot_data[2][0]
        #     ISI_diff_mean = np.mean(ISI_diff)
        #
        #     AA = [[]] * 8
        #     # ISI, SYNC, DUR, COUNT
        #     AA[0] = plot_data[2][0]
        #     AA[1] = plot_data[2][1]
        #     AA[2] = plot_data[2][2] / np.max(plot_data[2][2])
        #     AA[3] = plot_data[2][3] / np.max(plot_data[2][3])
        #
        #     # Pulse trains
        #     AA[4] = plot_data[3][0][dur_d]
        #     AA[5] = plot_data[3][1][dur_d]
        #     AA[6] = plot_data[3][2][dur_d] / np.max(plot_data[3][2][dur_d])
        #     AA[7] = plot_data[3][3][dur_d] / np.max(plot_data[3][3][dur_d])
        #
        #     cors = np.zeros((len(AA), len(AA)))
        #     for jj in range(len(AA)):
        #         for kk in range(len(AA)):
        #             dummy = scipy.signal.correlate2d(AA[jj], AA[kk], mode='full')
        #             cors[jj, kk] = np.max(dummy)
        #
        #     # c = 0.001
        #     BB_S = (AA[0] + (1-AA[1]) + AA[2] + AA[3]) / 4
        #     BB_P = (AA[4] + (1-AA[5]) + AA[6] + AA[7]) / 4
        #
        #     plt.subplot(121)
        #     plt.imshow(BB_S)
        #     plt.colorbar()
        #     plt.subplot(122)
        #     plt.imshow(BB_P)
        #     plt.colorbar()
        #     # cors = cors / np.max(cors)
        #     # # LINEAR COMBINATION OPT.
        #     # # ww = np.array([0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1])
        #     # ww = np.array(np.linspace(0.01, 1, 10))
        #     # var = np.zeros(len(ww))
        #     # result = np.zeros((len(ww), len(ww), len(ww), len(ww)))
        #     # for w_count in range(len(ww)):
        #     #     for w_sync in range(len(ww)):
        #     #         for w_dur in range(len(ww)):
        #     #             for w_isi in range(len(ww)):
        #     #                 # w = [ww[w_isi], ww[w_dur], ww[w_sync], 0]
        #     #                 # ISI_weight = w[0] * ISI_S + w[1] * (1/SYNC_S) + w[2] * DUR_S + w[3] * COUNT_S
        #     #                 ISI_weight = ww[w_isi] * ISI_S + ww[w_sync] * (1/SYNC_S) + ww[w_dur] * DUR_S + ww[w_count] * COUNT_S
        #     #                 ISI_weight_norm = ISI_weight / np.max(ISI_weight)
        #     #                 ISI_mins = np.zeros(len(ISI_weight_norm))
        #     #                 cr = 0
        #     #                 for q in range(len(ISI_weight_norm)):
        #     #                     # ISI_mins[q] = np.where(ISI_weight_norm[q, :] == np.min(ISI_weight_norm[q, :]))
        #     #                     mm = np.where(ISI_weight_norm[q, :] == np.min(ISI_weight_norm[q, :]))[0][0]
        #     #                     if mm == q:
        #     #                         cr += 1
        #     #                 result[w_count, w_sync, w_dur, w_isi] = cr
        #     #     # fig, ax = plt.subplots()
        #     #     # ax.imshow(ISI_weight_norm)
        #     # # plt.show()
        #     # a = np.where(result == np.max(result))
        #     # b = [[]] * len(a[0])
        #     # from mpl_toolkits.mplot3d import Axes3D
        #     # fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
        #     # ax.scatter(b[kk][0], b[kk][2], b[kk][1], zdir='z', c='red')
        #     #
        #     # for kk in range(len(a[0])):
        #     #     b[kk] = [ww[a[0][kk]], ww[a[1][kk]], ww[a[2][kk]], ww[a[3][kk]]]
        #     #     ax.scatter(b[kk][0], b[kk][2], b[kk][1], zdir='z', c='red')
        #     # ax.set_xlabel('COUNT weight')
        #     # ax.set_ylabel('DUR weight')
        #     # ax.set_zlabel('SYNC weight')
        #     # plt.show()
        #     embed()
        #     exit()
        # Plot on axes

        x = np.arange(0, len(COUNT_S)+1, 1)
        y = np.arange(0, len(COUNT_S)+1, 1)
        XX, YY = np.meshgrid(x, y)
        l_width = 0.2
        edge_color = 'k'
        im1 = ax1.pcolormesh(XX, YY, plot_data[2][0], cmap='viridis', vmin=0, vmax=1, shading='flat', edgecolor=edge_color, linewidths=l_width)
        im2 = ax2.pcolormesh(XX, YY, plot_data[2][1], vmin=0, vmax=1, cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)
        im3 = ax3.pcolormesh(XX, YY, plot_data[2][2], vmin=c3_settings[0], vmax=c3_settings[-1], cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)
        im4 = ax4.pcolormesh(XX, YY, COUNT_S, vmin=c4_settings[0], vmax=c4_settings[-1], cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)

        im5 = ax5.pcolormesh(XX, YY, plot_data[3][0][dur_d], vmin=0, vmax=1, cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)
        im6 = ax6.pcolormesh(XX, YY, plot_data[3][1][dur_d], vmin=0, vmax=1, cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)
        im7 = ax7.pcolormesh(XX, YY, plot_data[3][2][dur_d], vmin=c3_settings[0], vmax=c3_settings[-1], cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)
        im8 = ax8.pcolormesh(XX, YY, COUNT_P, vmin=c4_settings[0], vmax=c4_settings[-1], cmap='viridis', shading='flat', edgecolor=edge_color, linewidths=l_width)


        # im1 = ax1.imshow(plot_data[2][0], vmin=0, vmax=1, cmap='viridis')
        # im2 = ax2.imshow(plot_data[2][1], vmin=0, vmax=1, cmap='viridis')
        # im3 = ax3.imshow(plot_data[2][2], vmin=c3_settings[0], vmax=c3_settings[-1], cmap='viridis')
        # im4 = ax4.imshow(COUNT_S, vmin=c4_settings[0], vmax=c4_settings[-1], cmap='viridis')

        # im5 = ax5.imshow(plot_data[3][0][dur_d], vmin=0, vmax=1, cmap='viridis')
        # im6 = ax6.imshow(plot_data[3][1][dur_d], vmin=0, vmax=1, cmap='viridis')
        # im7 = ax7.imshow(plot_data[3][2][dur_d], vmin=c3_settings[0], vmax=c3_settings[-1], cmap='viridis')
        # im8 = ax8.imshow(COUNT_P, vmin=c4_settings[0], vmax=c4_settings[-1], cmap='viridis')

        c1 = matplotlib.colorbar.ColorbarBase(cb1, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1), orientation='horizontal', ticklocation='top')
        c1.set_label(cbar_labels[0])
        c1.set_ticks([0, 0.5, 1])
        c1.set_ticklabels([0, 0.5, 1])

        c2 = matplotlib.colorbar.ColorbarBase(cb2, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1), orientation='horizontal', ticklocation='top')
        c2.set_label(cbar_labels[1])
        c2.set_ticks([0, 0.5, 1])
        c2.set_ticklabels([0, 0.5, 1])

        c3 = matplotlib.colorbar.ColorbarBase(cb3, cmap='viridis',
                                              norm=matplotlib.colors.Normalize(vmin=c3_settings[0], vmax=c3_settings[-1]), orientation='horizontal', ticklocation='top')
        c3.set_label(cbar_labels[2])
        c3.set_ticks(c3_settings)
        c3.set_ticklabels(c3_settings)

        c4 = matplotlib.colorbar.ColorbarBase(cb4, cmap='viridis',
                                              norm=matplotlib.colors.Normalize(vmin=c4_settings[0], vmax=c4_settings[-1]), orientation='horizontal', ticklocation='top')
        c4.set_label(cbar_labels[3])
        c4.set_ticks(c4_settings)
        c4.set_ticklabels(c4_settings)

        ax1.set_xlim(0, len(COUNT_S))
        ax1.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax1.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax1.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax2.set_xlim(0, len(COUNT_S))
        ax2.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax2.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax2.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax3.set_xlim(0, len(COUNT_S))
        ax3.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax3.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax3.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax4.set_xlim(0, len(COUNT_S))
        ax4.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax4.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax4.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax5.set_xlim(0, len(COUNT_S))
        ax5.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax5.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax5.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax6.set_xlim(0, len(COUNT_S))
        ax6.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax6.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax6.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax7.set_xlim(0, len(COUNT_S))
        ax7.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax7.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax7.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax8.set_xlim(0, len(COUNT_S))
        ax8.set_xticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax8.set_xticklabels(np.arange(0, len(COUNT_S), 5))
        ax8.set_xticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        # Y Axis
        ax1.set_ylim(len(COUNT_S), 0)
        ax1.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax1.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax1.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax2.set_ylim(len(COUNT_S), 0)
        ax2.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax2.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax2.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax3.set_ylim(len(COUNT_S), 0)
        ax3.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax3.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax3.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax4.set_ylim(len(COUNT_S), 0)
        ax4.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax4.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax4.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax5.set_ylim(len(COUNT_S), 0)
        ax5.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax5.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax5.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax6.set_ylim(len(COUNT_S), 0)
        ax6.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax6.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax6.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax7.set_ylim(len(COUNT_S), 0)
        ax7.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax7.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax7.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        ax8.set_ylim(len(COUNT_S), 0)
        ax8.set_yticks(np.arange(0, len(COUNT_S), 5) + 0.5)
        ax8.set_yticklabels(np.arange(0, len(COUNT_S), 5))
        ax8.set_yticks(np.arange(0, len(COUNT_S), 1) + 0.5, minor=True)

        # ax1.tick_params(axis='both', which='minor', colors='white')

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])

        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax6.set_yticklabels([])
        ax7.set_yticklabels([])
        ax8.set_yticklabels([])

        # Remove axis lines
        sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax3, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax4, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax5, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax6, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax7, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
        sns.despine(ax=ax8, top=True, right=True, left=True, bottom=True, offset=False, trim=False)

        # # Grid
        # grid_color = 'w'
        # grid_linewidth = 1
        # ax1.grid(which='minor', color=grid_color, linestyle='-', linewidth=grid_linewidth, snap=True)
        # ax2.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax3.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax4.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax5.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax6.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax7.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)
        # ax8.grid(which='both', color=grid_color, linestyle='-', linewidth=grid_linewidth)

        # X, Y = np.meshgrid(np.arange(0, len(COUNT_P), 1), np.arange(0, len(COUNT_P), 1))
        # ax1.contourf(X, Y)

        # Subfig caps
        subfig_caps = 12
        label_x_pos = -0.2
        label_y_pos = 0.92
        subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        ax1.text(label_x_pos-0.15, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps, color='black')
        # sfc1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
        ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps, color='black')
        ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps, color='black')
        ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps, color='black')
        ax5.text(label_x_pos-0.15, label_y_pos, subfig_caps_labels[4], transform=ax5.transAxes, size=subfig_caps, color='black')
        ax6.text(label_x_pos, label_y_pos, subfig_caps_labels[5], transform=ax6.transAxes, size=subfig_caps, color='black')
        ax7.text(label_x_pos, label_y_pos, subfig_caps_labels[6], transform=ax7.transAxes, size=subfig_caps, color='black')
        ax8.text(label_x_pos, label_y_pos, subfig_caps_labels[7], transform=ax8.transAxes, size=subfig_caps, color='black')

        # Axes Labels
        fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
        fig.text(0.045, 0.60, 'Matched call', ha='center', fontdict=None, rotation=90)
        fig.text(0.92, 0.38, 'Pulse trains', ha='center', fontdict=None, rotation=-90)
        fig.text(0.92, 0.73, 'Spike trains', ha='center', fontdict=None, rotation=-90)

        fig.subplots_adjust(left=0.12, top=0.9, bottom=0.12, right=0.9, wspace=0.1, hspace=0.1)
        # figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/' + stim_type + '_' + str(
        #     selected_duration) + '_comparison.pdf'

        # figname = "/media/nils/Data/Moth/figs/" + data_name + '/' + stim_type + '_' + str(
        #     selected_duration) + '_comparison.pdf'
        figname = path_names[2] + stim_type + '_' + str(selected_duration) + '_comparison.pdf'
        fig.savefig(figname)

    exit()
    for aa in range(4):
        if method[aa] is 'vr':
            fig = plt.figure(figsize=(5.9, 3.9))
            grid = matplotlib.gridspec.GridSpec(nrows=23, ncols=25)
            ax1 = plt.subplot(grid[0:11, 0:8])
            ax2 = plt.subplot(grid[0:11, 8:16])
            ax3 = plt.subplot(grid[0:11, 16:24])
            ax4 = plt.subplot(grid[11:22, 0:8])
            ax5 = plt.subplot(grid[11:22, 8:16])
            ax6 = plt.subplot(grid[11:22, 16:24])
            ax_cbar = plt.subplot(grid[0:22, -1])

            im1 = ax1.imshow(plot_data[aa][0], vmin=0, vmax=np.max(plot_data[aa][0]), cmap='viridis')
            im2 = ax2.imshow(plot_data[aa][1], vmin=0, vmax=np.max(plot_data[aa][1]), cmap='viridis')
            im3 = ax3.imshow(plot_data[aa][2], vmin=0, vmax=np.max(plot_data[aa][2]), cmap='viridis')
            im4 = ax4.imshow(plot_data[aa][3], vmin=0, vmax=np.max(plot_data[aa][3]), cmap='viridis')
            im5 = ax5.imshow(plot_data[aa][4], vmin=0, vmax=np.max(plot_data[aa][4]), cmap='viridis')
            im6 = ax6.imshow(plot_data[aa][5], vmin=0, vmax=np.max(plot_data[aa][5]), cmap='viridis')
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap='viridis')
            cb1.set_label('Normalized Van Rossum Distance')


            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xticks(np.arange(0, 20, 5))
            ax5.set_xticks(np.arange(0, 20, 5))
            ax6.set_xticks(np.arange(0, 20, 5))
            ax2.set_yticks([])
            ax3.set_yticks([])
            ax5.set_yticks([])
            ax6.set_yticks([])
            ax1.set_yticks(np.arange(0, 20, 5))
            ax4.set_yticks(np.arange(0, 20, 5))

            # Axes Labels
            fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
            fig.text(0.05, 0.55, 'Matched call', ha='center', fontdict=None, rotation=90)

            # Subfig caps
            subfig_caps = 12
            label_x_pos = 0.85
            label_y_pos = 0.85
            subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                     color='white')
            ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                     color='white')
            ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
                     color='white')
            ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
                     color='white')
            ax5.text(label_x_pos, label_y_pos, subfig_caps_labels[4], transform=ax5.transAxes, size=subfig_caps,
                     color='white')
            ax6.text(label_x_pos, label_y_pos, subfig_caps_labels[5], transform=ax6.transAxes, size=subfig_caps,
                     color='white')

            t1 = ax1.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[0]) + ' ms', transform=ax1.transAxes, size=6,
                     color=cc)
            t2 = ax1.text(0.1, 0.05, 'dur = ' + str(duration_pulses[0]) + ' ms', transform=ax1.transAxes, size=6,
                     color=cc)
            t3 = ax2.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[1]) + ' ms', transform=ax2.transAxes, size=6,
                     color=cc)
            t4 = ax2.text(0.1, 0.05, 'dur = ' + str(duration_pulses[1]) + ' ms', transform=ax2.transAxes, size=6,
                     color=cc)
            t5 = ax3.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[2]) + ' ms', transform=ax3.transAxes, size=6,
                     color=cc)
            t6 = ax3.text(0.1, 0.05, 'dur = ' + str(duration_pulses[2]) + ' ms', transform=ax3.transAxes, size=6,
                     color=cc)
            t7 = ax4.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[3]) + ' ms', transform=ax4.transAxes, size=6,
                     color=cc)
            t8 = ax4.text(0.1, 0.05, 'dur = ' + str(duration_pulses[3]) + ' ms', transform=ax4.transAxes, size=6,
                     color=cc)
            t9 = ax5.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[4]) + ' ms', transform=ax5.transAxes, size=6,
                     color=cc)
            t10 = ax5.text(0.1, 0.05, 'dur = ' + str(duration_pulses[4]) + ' ms', transform=ax5.transAxes, size=6,
                     color=cc)
            t11 = ax6.text(0.1, 0.1, r'$\tau$ = ' + str(taus_pulses[5]) + ' ms', transform=ax6.transAxes, size=6,
                     color=cc)
            t12 = ax6.text(0.1, 0.05, 'dur = ' + str(duration_pulses[5]) + ' ms', transform=ax6.transAxes, size=6,
                     color=cc)

            t1.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t2.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t3.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t4.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t5.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t6.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t7.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t8.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t9.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t10.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t11.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
            t12.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

            fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.2, hspace=0.4)
            figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + plot_name[aa] + stim_type + '.pdf'
            fig.savefig(figname)
            plt.close(fig)

        else:
            fig = plt.figure(figsize=(5.9, 4.9))
            grid = matplotlib.gridspec.GridSpec(nrows=82, ncols=102)
            ax1 = plt.subplot(grid[0:40, 0:40])
            cb1 = plt.subplot(grid[1:39, 41])

            ax2 = plt.subplot(grid[0:40, 60:100])
            cb2 = plt.subplot(grid[1:39, 101])

            ax3 = plt.subplot(grid[41:81, 0:40])
            cb3 = plt.subplot(grid[42:80, 41])

            ax4 = plt.subplot(grid[41:81, 60:100])
            cb4 = plt.subplot(grid[42:80, 101])

            if method[aa] is 'sd':
                im1 = ax1.imshow(plot_data[aa][0], vmin=0, vmax=1, cmap='viridis')
                im2 = ax2.imshow(plot_data[aa][1], vmin=0, vmax=1, cmap='viridis')
                im3 = ax3.imshow(plot_data[aa][2], vmin=0, vmax=np.max(plot_data[aa][2]), cmap='viridis')
                im4 = ax4.imshow(plot_data[aa][3], vmin=0, vmax=np.max(plot_data[aa][3]), cmap='viridis')
            else:
                im1 = ax1.imshow(plot_data[aa][0][dur_d], vmin=0, vmax=1, cmap='viridis')
                im2 = ax2.imshow(plot_data[aa][1][dur_d], vmin=0, vmax=1, cmap='viridis')
                im3 = ax3.imshow(plot_data[aa][2][dur_d], vmin=0, vmax=np.max(plot_data[aa][2]), cmap='viridis')
                im4 = ax4.imshow(plot_data[aa][3][dur_d], vmin=0, vmax=np.max(plot_data[aa][3]), cmap='viridis')

            c1 = matplotlib.colorbar.ColorbarBase(cb1, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
            c2 = matplotlib.colorbar.ColorbarBase(cb2, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
            c3 = matplotlib.colorbar.ColorbarBase(cb3, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(plot_data[aa][2])))
            c4 = matplotlib.colorbar.ColorbarBase(cb4, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(plot_data[aa][3])))
            c1.set_label(cbar_labels[0])
            c2.set_label(cbar_labels[1])
            c3.set_label(cbar_labels[2])
            c4.set_label(cbar_labels[3])

            # axes
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks(np.arange(0, 20, 5))
            ax4.set_xticks(np.arange(0, 20, 5))
            ax4.set_yticks([])
            ax1.set_yticks(np.arange(0, 20, 5))
            ax3.set_yticks(np.arange(0, 20, 5))
            ax2.set_yticks([])

            # Axes Labels
            fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)
            fig.text(0.05, 0.55, 'Matched call', ha='center', fontdict=None, rotation=90)

            # Subfig caps
            subfig_caps = 12
            label_x_pos = 0.85
            label_y_pos = 0.85
            subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps, color='white')
            ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps, color='white')
            ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps, color='white')
            ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps, color='white')

            # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.5, hspace=0.1)
            figname = "/media/brehm/Data/MasterMoth/figs/" + data_name + plot_name[aa] + stim_type + '_' + str(selected_duration) +'.pdf'
            fig.savefig(figname)
            plt.close(fig)

if FI_OVERANIMALS:
    # Load data
    # all_data = [freqs, estimated_th_conv, estimated_th_d, estimated_th_r, estimated_th_inst]
    save_fig = True
    species = 'Carales'

    # if species is 'Estigmene':
        # Estigmene:
    # datasets_20_estigmene = ['2017-11-25-aa', '2017-11-27-aa']  # 20 ms
    datasets_20_estigmene = ['2017-11-25-aa', '2017-11-27-aa', '2017-10-26-aa', '2017-12-05-aa']  # 20 ms
    # datasets_50 = ['2017-10-26-aa', '2017-12-05-aa']  # 50 ms
    # elif species is 'Carales':
        # Carales:

    datasets_20_carales = ['2017-11-01-aa', '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']  # 20 ms

    # datasets_50 = ['2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac']  # 50 ms
    # datasets_05 = ['2017-10-23-ah']  # 5 ms

    # Load calls
    fs = 480 * 1000
    # carales_calls = ['carales_11x11_01', 'carales_11x11_02', 'carales_12x12_01', 'carales_12x12_02', 'carales_13x13_01', 'carales_13x13_02', 'carales_19x19']
    carales_calls = ['carales_12x12_01', 'carales_12x12_02']

    estigmene = wav.read('/media/brehm/Data/MasterMoth/Esitgmene/Pk13060008_m.wav')[1]
    y = estigmene[835000:996000]
    freqs_estigmene, power_estigmene = signal.welch(y, fs, scaling='spectrum')
    freqs_estigmene = freqs_estigmene / 1000

    # bat = wav.read('/media/brehm/Data/MasterMoth/stimuli_backup/batcalls/Myotis_bechsteinii_1_n.wav')
    # y_bat = bat[1]
    # fs_bat = bat[0]
    # freqs_bat, power_bat = signal.welch(y_bat, fs_bat, scaling='spectrum')
    # freqs_bat = freqs_bat / 1000

    carales = [[]] * len(carales_calls)
    freqs = [[]] * len(carales_calls)
    power = [[]] * len(carales_calls)

    for j in range(len(carales_calls)):
        carales[j] = wav.read('/media/brehm/Data/MasterMoth/stimuli_backup/naturalmothcalls/' + carales_calls[j] + '.wav')[1]
        x = carales[j]
        # plt.plot(x)
        freqs[j], power[j] = signal.welch(x, fs, scaling='spectrum')
        freqs[j] = freqs[j] / 1000

    bat_calls = ['Barbastella_barbastellus_1_n', 'Eptesicus_nilssonii_1_s', 'Myotis_bechsteinii_1_n',
                 'Myotis_brandtii_1_n', 'Myotis_nattereri_1_n', 'Nyctalus_leisleri_1_n', 'Nyctalus_noctula_2_s',
                 'Pipistrellus_pipistrellus_1_n', 'Pipistrellus_pygmaeus_2_n', 'Rhinolophus_ferrumequinum_1_n',
                 'Vespertilio_murinus_1_s']
    bats = [[]] * len(bat_calls)
    freqs_bats = [[]] * len(bat_calls)
    power_bats = [[]] * len(bat_calls)
    for j in range(len(bat_calls)):
        bats[j] = wav.read('/media/brehm/Data/MasterMoth/stimuli_backup/batcalls/' + bat_calls[j] + '.wav')
        fs = bats[j][0]
        x = bats[j][1]
        # plt.plot(x)
        freqs_bats[j], power_bats[j] = signal.welch(x, fs, scaling='spectrum')
        freqs_bats[j] = freqs_bats[j] / 1000
    power_bats_mean = np.median(np.array(power_bats), axis=0)
    power_bats_std = np.std(np.array(power_bats), axis=0)
    power_bats_mean[freqs_bats[0] <= 5] = 0

    # plt.show()
    # mean_power = [[]] * 3
    # mean_power[0] = np.mean(np.array(power[:2]), axis=0)
    # mean_power[1] = np.mean(np.array(power[2:4]), axis=0)
    # mean_power[2] = np.mean(np.array(power[4:6]), axis=0)

    mean_power = np.mean(np.array(power), axis=0)
    std_power = np.std(np.array(power), axis=0)
    error_up = mean_power + std_power
    error_down = mean_power - std_power
    idx = error_down < 0
    error_down[idx] = 1

    # plt.semilogy(freqs, power)
    # plt.xlim(0, 100)
    # plt.xticks(np.arange(0, 100, 20))
    # plt.show()
    # f, t, Sxx = signal.spectrogram(x, fs, nfft=512, window=signal.get_window('hamming', 250), noverlap=200)
    # plt.pcolormesh(t, f, Sxx, cmap='Spectral', vmin=100, vmax=1500)
    # plt.colorbar()
    # plt.show()
    # f, Pxx_den = signal.periodogram(x, fs, window=signal.get_window('hamming', 250), nfft=512, return_onesided=True, scaling='spectrum', axis=-1)
    # plt.specgram(carales[1], NFFT=256, Fs=2, Fc=0, window=np.hanning, noverlap=128, cmap=None, xextent=None, pad_to=None, sides='default', scale_by_freq=None, mode='default', scale='default')
    # plt.specgram(carales[1],  NFFT=256, Fs=480*1000, Fc=0, window=np.hanning, noverlap=128)

    fi_20_estigmene = [[]] * len(datasets_20_estigmene)
    for i in range(len(datasets_20_estigmene)):
        data_name = datasets_20_estigmene[i]
        path_names = mf.get_directories(data_name=data_name)
        p = path_names[1]
        with open(p + 'fi_field.txt', 'rb') as fp:  # Unpickling
            all_data_20_estigmene = pickle.load(fp)
            fi_20_estigmene[i] = all_data_20_estigmene

    fi_20_carales = [[]] * len(datasets_20_carales)
    for i in range(len(datasets_20_carales)):
        data_name = datasets_20_carales[i]
        path_names = mf.get_directories(data_name=data_name)
        p = path_names[1]
        with open(p + 'fi_field.txt', 'rb') as fp:  # Unpickling
            all_data_20_carales = pickle.load(fp)
            fi_20_carales[i] = all_data_20_carales

    # Plot
    mf.plot_settings()
    subfig_caps = 12
    fig = plt.figure()
    ax1_power = plt.subplot(321)
    ax2_power = plt.subplot(322)
    ax1 = plt.subplot(323)
    ax2 = plt.subplot(324)
    ax3 = plt.subplot(325)
    ax4 = plt.subplot(326)

    # Plot Power Spectrum of calls
    # ax1_power.fill_between(freqs[0], mean_power-std_power, mean_power+std_power, facecolors='k', alpha=0.5)
    ax1_power.semilogy(freqs_bats[0], power_bats_mean, 'k:', label='Bats')
    ax2_power.semilogy(freqs_bats[0], power_bats_mean, 'k:', label='Bats')

    # ax1_power.fill_between(freqs[0], power_bats_mean-power_bats_std, power_bats_mean+power_bats_std, facecolors='r', alpha=0.5)

    ax1_power.semilogy(freqs_estigmene, power_estigmene, 'k', label='Estigmene')
    ax2_power.fill_between(freqs[0], mean_power-std_power, mean_power+std_power, facecolors='k', alpha=0.5)
    ax2_power.semilogy(freqs[0], mean_power, 'k', label='Carales')
    ax1_power.set_ylabel('Power')
    ax1_power.legend(frameon=False)
    ax2_power.legend(frameon=False)

    for k in range(len(fi_20_estigmene)):
        if k == 0:
            ax1.plot(fi_20_estigmene[k][0], fi_20_estigmene[k][1], '-', label='20 ms', color='k')
        elif k == 1:
            ax1.plot(fi_20_estigmene[k][0], fi_20_estigmene[k][1], '-', color='k')
        elif k == 2:
            ax1.plot(fi_20_estigmene[k][0], fi_20_estigmene[k][1], '--', label='50 ms', color='k')
        else:
            ax1.plot(fi_20_estigmene[k][0], fi_20_estigmene[k][1], '--', color='k')

    for k in range(len(fi_20_carales)):
        if k == 0:
            ax2.plot(fi_20_carales[k][0], fi_20_carales[k][1], '-', label='Carales - 20 ms', color='k')
        else:
            ax2.plot(fi_20_carales[k][0], fi_20_carales[k][1], '-', color='k')

    lin_styles = ['-', '--', ':', '-.']
    cc = ['0.5', 'k', 'k', 'k']
    method_labels = ['convolved rate', 'SYNC', 'spike count', 'instantaneous rate']
    for j in range(3):
        ax3.plot(fi_20_estigmene[0][0], fi_20_estigmene[0][j+1], lin_styles[j], color=cc[j], label=method_labels[j])
        ax4.plot(fi_20_carales[0][0], fi_20_carales[0][j + 1], lin_styles[j], color=cc[j], label=method_labels[j])

    # Subplot Letters
    label_x_pos = -0.2
    label_y_pos = 1.1
    ax1_power.text(label_x_pos, label_y_pos, 'a', transform=ax1_power.transAxes, size=subfig_caps)
    ax2_power.text(label_x_pos, label_y_pos, 'b', transform=ax2_power.transAxes, size=subfig_caps)
    ax1.text(label_x_pos, label_y_pos, 'c', transform=ax1.transAxes, size=subfig_caps)
    ax2.text(label_x_pos, label_y_pos, 'd', transform=ax2.transAxes, size=subfig_caps)
    ax3.text(label_x_pos, label_y_pos, 'e', transform=ax3.transAxes, size=subfig_caps)
    ax4.text(label_x_pos, label_y_pos, 'f', transform=ax4.transAxes, size=subfig_caps)

    ax1.legend(frameon=False)
    ax3.legend(frameon=False)
    ax4.legend(frameon=False)
    y_min = 0
    y_max = 100
    y_step = 20
    x_min = 0
    x_max = 100
    x_step = 20

    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(np.arange(y_min, y_max+y_step, y_step))
    ax2.set_ylim(y_min, y_max)
    ax2.set_yticks(np.arange(y_min, y_max+y_step, y_step))
    ax3.set_ylim(y_min, y_max)
    ax3.set_yticks(np.arange(y_min, y_max+y_step, y_step))
    ax4.set_ylim(y_min, y_max)
    ax4.set_yticks(np.arange(y_min, y_max+y_step, y_step))

    ax1_power.set_ylim([1e0, 1e6])
    ax2_power.set_ylim([1e0, 1e6])
    ax1_power.set_xlim(x_min, x_max)
    ax1_power.set_xticks(np.arange(x_min, x_max+x_step, x_step))
    ax2_power.set_xlim(x_min, x_max)
    ax2_power.set_xticks(np.arange(x_min, x_max + x_step, x_step))

    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(np.arange(x_min, x_max+x_step, x_step))
    ax2.set_xlim(x_min, x_max)
    ax2.set_xticks(np.arange(x_min, x_max+x_step, x_step))
    ax3.set_xlim(x_min, x_max)
    ax3.set_xticks(np.arange(x_min, x_max+x_step, x_step))
    ax4.set_xlim(x_min, x_max)
    ax4.set_xticks(np.arange(x_min, x_max+x_step, x_step))

    ax1.set_ylabel('Intensity [dB SPL]')
    ax3.set_ylabel('Intensity [dB SPL]')
    fig.text(0.5, 0.035, 'Frequency [kHz]', ha='center', fontdict=None)
    sns.despine()

    # Save Plot to HDD
    p = "/media/brehm/Data/MasterMoth/figs/"
    figname = p + 'fi_field.pdf'
    fig.set_size_inches(5.9, 5.9)
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.4, hspace=0.4)
    fig.savefig(figname)
    plt.close(fig)
    print('Plot saved')

if OVERALLVS:
    # RectInvterval
    # datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2017-12-05-ab', '2018-02-16-aa']
    datasets = ['2017-11-27-aa', '2017-11-29-aa', '2018-02-16-aa']  # all Estigmene female

    # MAS
    # [female, female, male, male, male, male]
    # datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-01-ab', '2017-12-01-ac', '2017-12-05-ab']
    # [female, female, male]
    # datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-05-ab']  # all Estigmene

    protocol_name = 'PulseIntervalsRect'
    # protocol_name = 'intervals_mas'
    vs = [[]] * len(datasets)
    mf.plot_settings()
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    marks = ['o', 's', 'v', '>', 'd']
    if protocol_name is 'PulseIntervalsRect':
        cut1 = 17
        cut2 = 34
    if protocol_name is 'intervals_mas':
        cut1 = 16
        cut2 = 32
    label1 = [None] * len(datasets)
    label1[0] = 'n.s.'
    label2 = [None] * len(datasets)
    label2[0] = 'sig.'

    for k in range(len(datasets)):
        data_name = datasets[k]
        path_names = mf.get_directories(data_name=data_name)
        p = path_names[1]
        vs[k] = np.load(p + protocol_name + '_vs.npy')

        vs_mean = vs[k][0:cut1, 3]
        gaps = vs[k][0:cut1, 2]
        ci_low = vs[k][0:cut1, 4]
        ci_up = vs[k][0:cut1, 5]
        percentile = vs[k][0:cut1, 7]
        low = vs_mean - ci_low
        up = ci_up - vs_mean
        # Find Threshold
        idx = vs_mean >= percentile
        gap_th = np.min(gaps[idx])
        gap_vs = vs_mean[gap_th == gaps]
        ax1.errorbar(gaps, vs_mean, yerr=[up, low], label=label1[k], color='k', marker=marks[k], linestyle='-', markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.5, markersize=3)
        ax1.errorbar(gaps[idx], vs_mean[idx], yerr=[up[idx], low[idx]], label=label2[k], color='k', marker=marks[k], linestyle='-', markersize=3)

        vs_mean = vs[k][cut1:cut2, 3]
        gaps = vs[k][cut1:cut2, 2]
        ci_low = vs[k][cut1:cut2, 4]
        ci_up = vs[k][cut1:cut2, 5]
        percentile = vs[k][cut1:cut2, 7]
        low = vs_mean - ci_low
        up = ci_up - vs_mean
        # Find Threshold
        idx = vs_mean >= percentile
        gap_th = np.min(gaps[idx])
        gap_vs = vs_mean[gap_th == gaps]
        ax2.errorbar(gaps, vs_mean, yerr=[up, low], label=label1[k], color='k', marker=marks[k], linestyle='-',
                     markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.5, markersize=3)
        ax2.errorbar(gaps[idx], vs_mean[idx], yerr=[up[idx], low[idx]], label=label2[k], color='k', marker=marks[k],
                     linestyle='-', markersize=3)

        # ax2.plot(gap_th, gap_vs, 'ro')

    ax1.set_xlim(-1, 21)
    ax1.set_xticks(np.arange(0, 20 + 2, 5))
    ax2.set_xlim(-1, 21)
    ax2.set_xticks(np.arange(0, 20 + 2, 5))
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    label_x_pos = -0.2
    label_y_pos = 1.1
    subfig_caps = 12
    ax1.text(label_x_pos, label_y_pos, 'a', transform=ax1.transAxes, size=subfig_caps)
    ax2.text(label_x_pos, label_y_pos, 'b', transform=ax2.transAxes, size=subfig_caps)
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax1.set_ylabel('Mean vector strength')
    fig.text(0.5, 0.075, 'Gap [ms]', ha='center', fontdict=None)
    sns.despine()

    # Save Plot to HDD
    p = "/media/brehm/Data/MasterMoth/figs/"
    figname = p + protocol_name + '_VS.pdf'
    fig.set_size_inches(5.9, 2.9)
    fig.subplots_adjust(left=0.1, top=0.8, bottom=0.2, right=0.9, wspace=0.4, hspace=0.4)
    fig.savefig(figname)
    plt.close(fig)
    print('Plot saved')

if PLOT_CORRS:
    protocol_name = 'PulseIntervalsRect'
    # good recordings Rect
    datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2018-02-16-aa']
    # corrs = [[]] * len(datasets)
    corrs = []
    for k in range(len(datasets)):
        data_name = datasets[k]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        p = path_names[1]
        dummy = np.load(p + protocol_name + '_corr.npy')
        corrs.append(dummy)

    # idx = 17
    # gaps = corrs[:idx, 0]
    # r_rate = corrs[:idx, 1]
    # r_sync = corrs[:idx, 2]
    # lag_rate = corrs[:idx, 3]
    # lag_sync = corrs[:idx, 4]

    # Plot
    # Prepare Axes
    mf.plot_settings()
    subfig_caps = 12
    fig = plt.figure()
    fig_size = (1, 2)
    fig.set_size_inches(5.9, 1.9)
    fig.subplots_adjust(left=0.15, top=0.8, bottom=0.2, right=0.9, wspace=0.4, hspace=0.4)

    corr_rate_ax = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=1)
    lag_rate_ax = plt.subplot2grid(fig_size, (0, 1), rowspan=1, colspan=1)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(lag_rate_ax,
                       width="50%",  # width = 30% of parent_bbox
                       height=0.4,  # height : 1 inch
                       loc='upper right')

    idx = 17
    for k in range(len(corrs)):
        gaps = corrs[k][:idx, 0]
        corr_rate_ax.plot(gaps, corrs[k][:idx, 1], '.-', label='Firing rate')

        lag_rate_ax.plot(gaps, corrs[k][:idx, 3] * 1000, '.-', label='Firing rate')

        axins.plot(gaps, corrs[k][:idx, 3] * 1000, '.-')

        # corr_sync_ax.plot(gaps, corrs[k][:idx, 2], 'k-', label='SYNC')
        # lag_sync_ax.plot(gaps, corrs[k][:idx, 4] * 1000, 'k-', label='SYNC')

    # corr_rate_ax.legend(frameon=False)
    # lag_rate_ax.legend(frameon=False)

    # Subplot Letters
    label_x_pos = -0.2
    label_y_pos = 1.1
    corr_rate_ax.text(label_x_pos, label_y_pos, 'a', transform=corr_rate_ax.transAxes, size=subfig_caps)
    lag_rate_ax.text(label_x_pos, label_y_pos, 'b', transform=lag_rate_ax.transAxes, size=subfig_caps)


    lag_lim1 = -300
    lag_lim2 = 500

    axins.set_ylim(-20, 20)
    axins.set_yticks(np.arange(-20, 21, 10))

    # axins.set_xlim(0, 21)
    axins.set_xticks(np.arange(0, 21, 5))
    axins.xaxis.set_tick_params(labelsize=6)
    axins.yaxis.set_tick_params(labelsize=6)
    # axins.set_xticklabels([])
    corr_rate_ax.set_ylim(0, 1)
    corr_rate_ax.set_yticks(np.arange(0, 1.1, 0.5))
    lag_rate_ax.set_ylim(lag_lim1, lag_lim2)
    lag_rate_ax.set_yticks([-300,-150, 0, 150, 300])

    # corr_sync_ax.set_ylim(0, 1)
    # corr_sync_ax.set_yticks(np.arange(0, 1.1, 0.5))
    # lag_sync_ax.set_ylim(lag_lim1, lag_lim2)
    # # lag_sync_ax.set_yticks([-150, -75, 0, 75, 150])

    # corr_ax.set_xlim(0, 1)
    corr_rate_ax.set_xticks(np.arange(0, 21, 5))
    # lag_ax.set_xlim(-150, 150)
    lag_rate_ax.set_xticks(np.arange(0, 21, 5))

    corr_rate_ax.set_ylabel('Max. Correlation')
    lag_rate_ax.set_ylabel('Lag [ms]')
    fig.text(0.5, 0.025, 'Gap [ms]', ha='center', fontdict=None)
    sns.despine()

    # Save Plot to HDD
    ps = "/media/brehm/Data/MasterMoth/figs/"
    figname = ps + protocol_name + '_corrs.pdf'
    fig.savefig(figname)
    plt.close(fig)

if PLOT_VR_TAUVSDUR_OVERALL:
    # taus = [1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
    # data_name = '2018-02-09-aa'
    # data_name = '2018-02-16-aa'
    # data_name = datasets[-1]
    datasets = ['2018-02-20-aa', '2018-02-16-aa', '2018-02-09-aa']
    vr_series = [[]] * len(datasets)
    vr_single = [[]] * len(datasets)

    for k in range(len(datasets)):
        data_name = datasets[k]
        path_names = mf.get_directories(data_name=data_name)
        p = path_names[1]

        vr_series[k] = np.load(p + 'VanRossum_correct_' + 'moth_series_selected' + '.npy')
        vr_single[k] = np.load(p + 'VanRossum_correct_' + 'moth_single_selected' + '.npy')

    vr_series_mean = np.mean(vr_series, axis=0)
    vr_series_std = np.std(vr_series, axis=0)
    vr_single_mean = np.mean(vr_single, axis=0)
    vr_single_std = np.std(vr_single, axis=0)

    mf.plot_settings()
    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=63)
    fig = plt.figure(figsize=(5.9, 2.3))
    ax1 = plt.subplot(grid[0:19])
    ax2 = plt.subplot(grid[21:40])
    ax3 = plt.subplot(grid[42:61])
    ax4 = plt.subplot(grid[62])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    # Image Plot
    x_single = list(np.arange(0, 255, 5))
    x_single[0] = 1
    x_series = list(np.arange(0, 2550, 50))
    x_series[0] = 10
    y = taus
    X_single, Y_single = np.meshgrid(x_single, y)
    X_series, Y_series = np.meshgrid(x_series, y)

    if stim_length is 'series':
        figname = '/media/brehm/Data/MasterMoth/figs/VanRossum_TauVSDur_series_overall.pdf'
        im1 = ax1.pcolormesh(X_series, Y_series, vr_series[0].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
        im2 = ax2.pcolormesh(X_series, Y_series, vr_series[1].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
        im3 = ax3.pcolormesh(X_series, Y_series, vr_series[2].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
    if stim_length is 'single':
        figname = '/media/brehm/Data/MasterMoth/figs/VanRossum_TauVSDur_single_overall.pdf'
        im1 = ax1.pcolormesh(X_single, Y_single, vr_single[0].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
        im2 = ax2.pcolormesh(X_single, Y_single, vr_single[1].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
        im3 = ax3.pcolormesh(X_single, Y_single, vr_single[2].T, cmap='jet', vmin=0, vmax=1, shading='gouraud')

    # im1 = ax1.pcolormesh(X_series, Y_series, vr_series_mean.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
    # im2 = ax2.pcolormesh(X_series, Y_series, vr_series_std.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')

    # im1 = ax1.pcolormesh(X_single, Y_single, vr_single_mean.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')
    # im2 = ax2.pcolormesh(X_single, Y_single, vr_single_std.T, cmap='jet', vmin=0, vmax=1, shading='gouraud')

    # grid[0].axhline(200, color='black', linestyle=':', linewidth=0.5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Axes Limits
    ax2.set_yticks([])
    ax3.set_yticks([])

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax4, cmap='jet', norm=norm)
    cb1.set_label('Correct')
    # cbar2 = plt.colorbar(im2, ticks=np.arange(0, 1.1, 0.2))
    # cbar2.ax.set_ylabel('Correct', rotation=270, labelpad=10)
    # cbar2.solids.set_rasterized(True)  # Removes white lines

    # Axes Labels
    ax1.set_ylabel('Tau [ms]')
    fig.text(0.5, 0.075, 'Spike train duration [ms]', ha='center', fontdict=None)

    # Subfig Caps
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                 color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                 color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')

    # fig.set_size_inches(5.9, 1.9)
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.25, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(figname)
    plt.close(fig)

if PLOT_CORRECT_OVERALL:
    # data_name = '2018-02-09-aa'
    # data_name = '2018-02-15-aa'
    # data_name = datasets[-1]
    datasets = ['2018-02-20-aa', '2018-02-16-aa', '2018-02-09-aa']
    matches = [[]] * len(datasets)
    rand_matches = [[]] * len(datasets)

    for kk in range(len(datasets)):
        data_name = datasets[kk]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        p = path_names[1]

        matches[kk] = np.load(p + 'distances_correct_' + stim_type + '.npy')
        rand_matches[kk] = np.load(p + 'distances_rand_correct_' + stim_type + '.npy')

    matches_mean = np.mean(matches, axis=0)
    matches_std = np.std(matches, axis=0) / np.sqrt(len(datasets))
    rand_matches_mean = np.mean(rand_matches, axis=0)
    rand_matches_std = np.std(rand_matches, axis=0) / np.sqrt(len(datasets))

    # Plot
    mf.plot_settings()
    # fig = plt.figure(figsize=(5.9, 3.9))
    fig, ax = plt.subplots()

    marks = ['.', 'o', 'v', 's', '.']
    cc = ['0', 'orangered', 'navy', 'teal', '0']
    cc_e = ['0.8', 'orangered', 'navy', 'teal', '0.8']
    styles = [':', '-', '-', '-', '--']
    for k in range(len(profs)):
        # ax.errorbar(duration, matches_mean, yerr=matches_std, marker=marks[k], label=profs[k], color=cc[k], linestyle=styles[k], markersize=3)
        ax.errorbar(duration, matches_mean[:, k], yerr=matches_std[:, k], marker='', color=cc[k],
                    linestyle='', ecolor=cc[k], alpha=0.5)
        ax.plot(duration, matches_mean[:, k], marker=marks[k], label=profs[k], color=cc[k], linestyle=styles[k], markersize=3)


    ax.plot(duration, rand_matches_mean[:, 1], 'k', linewidth=2, label='Random')
    # ax.errorbar(duration, rand_matches_mean[:, 1], yerr=rand_matches_std[:, 1], color='k', linewidth=2, label='Random')
    rand_mean = np.round(np.mean(rand_matches_mean[:, 1]), 2)
    # ax.text(2000, 0.1, 'random mean = ' + str(rand_mean), size=6, color='black')

    ax.set_xlabel('Spike train duration [ms]')
    ax.set_ylabel('Correct')
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(0, 1)
    if stim_length == 'series':
        ax.text(2000, 0.1, 'random mean = ' + str(rand_mean), size=6, color='black')
        ax.set_xticks(np.arange(0, duration[-1]+100, 500))
        ax.set_xlim(-0.2, duration[-1]+100)
    if stim_length == 'single':
        ax.text(200, 0.07, 'random mean = ' + str(rand_mean), size=6, color='black')
        ax.set_xticks(np.arange(0, duration[-1] + 10, 50))
        ax.set_xlim(-0.2, duration[-1] + 10)
    sns.despine()
    ax.legend(frameon=False)
    # Save Plot to HDD
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.4, hspace=0.4)
    fig.set_size_inches(5.9, 2.9)
    figname = '/media/brehm/Data/MasterMoth/figs/Distances_Correct_' + stim_type + '_overall.pdf'
    fig.savefig(figname)
    plt.close(fig)
    print('Distances Matrix Plot saved')

if PLOT_D_RATIOS_OVERALL:
    datasets = ['2018-02-20-aa', '2018-02-16-aa', '2018-02-09-aa']
    ratios_isi = [[]] * len(datasets)
    ratios_sync = [[]] * len(datasets)

    for kk in range(len(datasets)):
        data_name = datasets[kk]
        path_names = mf.get_directories(data_name=data_name)
        p = path_names[1]
        ratios_isi[kk] = np.load(p + 'DUR_Ratios_' + stim_type + '.npy')
        ratios_sync[kk] = np.load(p + 'COUNT_Ratios_' + stim_type + '.npy')

    ratios_isi_mean = np.mean(ratios_isi, axis=0)
    ratios_isi_std = np.std(ratios_isi, axis=0)
    ratios_sync_mean = np.mean(ratios_sync, axis=0)
    ratios_sync_std = np.std(ratios_sync, axis=0)

    # Normalize
    max_norm_isi = np.max(ratios_isi_mean[:, 2])
    ratios_isi_mean[:, 0] = ratios_isi_mean[:, 0] / max_norm_isi
    ratios_isi_mean[:, 1] = ratios_isi_mean[:, 1] / max_norm_isi
    ratios_isi_mean[:, 2] = ratios_isi_mean[:, 2] / max_norm_isi
    max_norm_sync = np.max(ratios_sync_mean[:, 2])
    ratios_sync_mean[:, 0] = ratios_sync_mean[:, 0] / max_norm_sync
    ratios_sync_mean[:, 1] = ratios_sync_mean[:, 1] / max_norm_sync
    ratios_sync_mean[:, 2] = ratios_sync_mean[:, 2] / max_norm_sync

    ratios_isi_std[:, 0] = ratios_isi_std[:, 0] / max_norm_isi
    ratios_isi_std[:, 1] = ratios_isi_std[:, 1] / max_norm_isi
    ratios_isi_std[:, 2] = ratios_isi_std[:, 2] / max_norm_isi
    ratios_sync_std[:, 0] = ratios_sync_std[:, 0] / max_norm_sync
    ratios_sync_std[:, 1] = ratios_sync_std[:, 1] / max_norm_sync
    ratios_sync_std[:, 2] = ratios_sync_std[:, 2] / max_norm_sync

    # Plot
    mf.plot_settings()
    if stim_length == 'series':
        x_end = 2500 + 100
        x_step = 500
    if stim_length == 'single':
        x_end = 250 + 10
        x_step = 50

    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    fig = plt.figure(figsize=(5.9, 2.9))
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1])
    ax3 = plt.subplot(grid[2])
    ax4 = plt.subplot(grid[3])

    ax1.plot(duration, ratios_isi_mean[:, 0], color='k', marker='', label='within')
    ax1.fill_between(duration, ratios_isi_mean[:, 0]-ratios_isi_mean[:, 1], ratios_isi_mean[:, 0]+ratios_isi_mean[:, 1], facecolors='k', alpha=0.25)
    # ax1.errorbar(duration, ratios_isi_mean[:, 0], yerr=ratios_isi_mean[:, 1], color='k', marker='', label='within')
    # ax1.errorbar(duration, ratios_isi_mean[:, 2], yerr=ratios_isi_std[:, 2], color='blue', marker='', label='between')
    ax1.fill_between(duration, ratios_isi_mean[:, 2]-ratios_isi_std[:, 2], ratios_isi_mean[:, 2]+ratios_isi_std[:, 2], facecolors='blue', alpha=0.25)
    ax1.plot(duration, ratios_isi_mean[:, 2], '-', label='between', color='blue')

    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.set_xticklabels([])
    ax1.set_ylabel('Norm. DUR')
    ax1.set_xlim(0, x_end)
    ax1.set_xticks(np.arange(0, x_end, x_step))

    ax3.plot(duration, ratios_isi_mean[:, 3], 'r-', label='ratio')
    ax3.fill_between(duration, ratios_isi_mean[:, 3]-ratios_isi_std[:, 3], ratios_isi_mean[:, 3]+ratios_isi_std[:, 3], facecolors='red', alpha=0.25)
    ax3.set_ylim(1, 3)
    ax3.set_yticks(np.arange(1, 3.1, 0.5))
    ax3.set_ylabel('Ratio')
    ax3.set_xlim(0, x_end)
    ax3.set_xticks(np.arange(0, x_end, x_step))

    ax2.plot(duration, ratios_sync_mean[:, 0], color='k', marker='', label='within')
    ax2.fill_between(duration, ratios_sync_mean[:, 0] - ratios_sync_mean[:, 1],
                     ratios_sync_mean[:, 0] + ratios_sync_mean[:, 1], facecolors='k', alpha=0.25)
    ax2.fill_between(duration, ratios_sync_mean[:, 2] - ratios_sync_std[:, 2],
                     ratios_sync_mean[:, 2] + ratios_sync_std[:, 2], facecolors='blue', alpha=0.25)
    ax2.plot(duration, ratios_sync_mean[:, 2], '-', label='between', color='blue')

    # ax2.errorbar(duration, ratios_sync_mean[:, 0], yerr=ratios_sync_mean[:, 1], color='k', marker='', label='within')
    # ax2.errorbar(duration, ratios_sync_mean[:, 2], yerr=ratios_sync_std[:, 2], color='blue', marker='', label='between')
    # # ax2.plot(duration, ratios_sync[:, 2], '-', label='between', color='blue')
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_ylabel('Nornm. COUNT')
    ax2.set_xlim(0, x_end)
    ax2.set_xticks(np.arange(0, x_end, x_step))

    ax4.plot(duration, ratios_sync_mean[:, 3], 'r-', label='ratio')
    ax4.fill_between(duration, (ratios_sync_mean[:, 3])-(ratios_sync_std[:, 3]), (ratios_sync_mean[:, 3])+(ratios_sync_std[:, 3]), facecolors='red', alpha=0.25)
    ax4.set_ylim(1, 3)
    ax4.set_yticks(np.arange(1, 3.1, 0.5))
    ax4.set_yticklabels([])
    ax4.set_xlim(0, x_end)
    ax4.set_xticks(np.arange(0, x_end, x_step))
    # ax4.set_ylabel('Ratio')


    # Axes Labels
    fig.text(0.5, 0.055, 'Spike train duration [ms]', ha='center', fontdict=None)

    # Subplot caps
    subfig_caps = 12
    label_x_pos = -0.15
    label_y_pos = 1.15
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                 color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                 color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')
    ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
             color='black')
    sns.despine()

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.4)
    figname = '/media/brehm/Data/MasterMoth/figs/Distance_Ratios_DUR_COUNT_' + stim_type + '_overall.pdf'
    fig.savefig(figname)
    plt.close(fig)

    # datasets = ['2018-02-20-aa', '2018-02-16-aa', '2018-02-09-aa']
    # ratios_isi = [[]] * len(datasets)
    # ratios_sync = [[]] * len(datasets)
    #
    # for kk in range(len(datasets)):
    #     data_name = datasets[kk]
    #     path_names = mf.get_directories(data_name=data_name)
    #     p = path_names[1]
    #     ratios_isi[kk] = np.load(p + 'ISI_Ratios_' + stim_type + '.npy')
    #     ratios_sync[kk] = np.load(p + 'SYNC_Ratios_' + stim_type + '.npy')
    #
    # ratios_isi_mean = np.mean(ratios_isi, axis=0)
    # ratios_isi_std = np.std(ratios_isi, axis=0)
    # ratios_sync_mean = np.mean(ratios_sync, axis=0)
    # ratios_sync_std = np.std(ratios_sync, axis=0)
    #
    # # Plot
    # mf.plot_settings()
    # if stim_length == 'series':
    #     x_end = 2500 + 100
    #     x_step = 500
    # if stim_length == 'single':
    #     x_end = 250 + 10
    #     x_step = 50
    #
    # # Create Grid
    # grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
    # fig = plt.figure(figsize=(5.9, 2.9))
    # ax1 = plt.subplot(grid[0])
    # ax2 = plt.subplot(grid[1])
    # ax3 = plt.subplot(grid[2])
    # ax4 = plt.subplot(grid[3])
    #
    # ax1.plot(duration, ratios_isi_mean[:, 0], color='k', marker='', label='within')
    # ax1.fill_between(duration, ratios_isi_mean[:, 0] - ratios_isi_mean[:, 1],
    #                  ratios_isi_mean[:, 0] + ratios_isi_mean[:, 1], facecolors='k', alpha=0.25)
    # # ax1.errorbar(duration, ratios_isi_mean[:, 0], yerr=ratios_isi_mean[:, 1], color='k', marker='', label='within')
    # # ax1.errorbar(duration, ratios_isi_mean[:, 2], yerr=ratios_isi_std[:, 2], color='blue', marker='', label='between')
    # ax1.fill_between(duration, ratios_isi_mean[:, 2] - ratios_isi_std[:, 2],
    #                  ratios_isi_mean[:, 2] + ratios_isi_std[:, 2], facecolors='blue', alpha=0.25)
    # ax1.plot(duration, ratios_isi_mean[:, 2], '-', label='between', color='blue')
    #
    # ax1.set_ylim(0, 1)
    # ax1.set_yticks(np.arange(0, 1.1, 0.2))
    # ax1.set_xticklabels([])
    # ax1.set_ylabel('ISI Distance')
    # ax1.set_xlim(0, x_end)
    # ax1.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax3.plot(duration, ratios_isi_mean[:, 3], 'r-', label='ratio')
    # ax3.fill_between(duration, ratios_isi_mean[:, 3] - ratios_isi_std[:, 3],
    #                  ratios_isi_mean[:, 3] + ratios_isi_std[:, 3], facecolors='red', alpha=0.25)
    # ax3.set_ylim(1, 3)
    # ax3.set_yticks(np.arange(1, 3.1, 0.5))
    # ax3.set_ylabel('Ratio')
    # ax3.set_xlim(0, x_end)
    # ax3.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax2.plot(duration, ratios_sync_mean[:, 0], color='k', marker='', label='within')
    # ax2.fill_between(duration, ratios_sync_mean[:, 0] - ratios_sync_mean[:, 1],
    #                  ratios_sync_mean[:, 0] + ratios_sync_mean[:, 1], facecolors='k', alpha=0.25)
    # ax2.fill_between(duration, ratios_sync_mean[:, 2] - ratios_sync_std[:, 2],
    #                  ratios_sync_mean[:, 2] + ratios_sync_std[:, 2], facecolors='blue', alpha=0.25)
    # ax2.plot(duration, ratios_sync_mean[:, 2], '-', label='between', color='blue')
    #
    # # ax2.errorbar(duration, ratios_sync_mean[:, 0], yerr=ratios_sync_mean[:, 1], color='k', marker='', label='within')
    # # ax2.errorbar(duration, ratios_sync_mean[:, 2], yerr=ratios_sync_std[:, 2], color='blue', marker='', label='between')
    # # # ax2.plot(duration, ratios_sync[:, 2], '-', label='between', color='blue')
    # ax2.set_ylim(0, 1)
    # ax2.set_yticks(np.arange(0, 1.1, 0.2))
    # ax2.set_xticklabels([])
    # ax2.set_yticklabels([])
    # ax2.set_ylabel('SYNC Value')
    # ax2.set_xlim(0, x_end)
    # ax2.set_xticks(np.arange(0, x_end, x_step))
    #
    # ax4.plot(duration, 1 / ratios_sync_mean[:, 3], 'r-', label='ratio')
    # ax4.fill_between(duration, (1 / ratios_sync_mean[:, 3]) - (ratios_sync_std[:, 3]),
    #                  (1 / ratios_sync_mean[:, 3]) + (ratios_sync_std[:, 3]), facecolors='red', alpha=0.25)
    # ax4.set_ylim(1, 3)
    # ax4.set_yticks(np.arange(1, 3.1, 0.5))
    # ax4.set_yticklabels([])
    # ax4.set_xlim(0, x_end)
    # ax4.set_xticks(np.arange(0, x_end, x_step))
    # # ax4.set_ylabel('Ratio')
    #
    #
    # # Axes Labels
    # fig.text(0.5, 0.055, 'Spike train duration [ms]', ha='center', fontdict=None)
    #
    # # Subplot caps
    # subfig_caps = 12
    # label_x_pos = -0.15
    # label_y_pos = 1.15
    # subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    # ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
    #          color='black')
    # ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
    #          color='black')
    # ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
    #          color='black')
    # ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
    #          color='black')
    # sns.despine()
    #
    # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.4)
    # figname = '/media/brehm/Data/MasterMoth/figs/Distance_Ratios_' + stim_type + '_overall.pdf'
    # fig.savefig(figname)
    # plt.close(fig)

if CALL_STATS:

    stims = ['naturalmothcalls/BCI1062_07x07.wav',
             'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav',
             'naturalmothcalls/carales_12x12_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
             'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/elysius_conspersus_11x11.wav',
             'naturalmothcalls/epidesma_oceola_06x06.wav',
             'naturalmothcalls/eucereon_appunctata_13x13.wav',
             'naturalmothcalls/eucereon_hampsoni_11x11.wav',
             'naturalmothcalls/eucereon_obscurum_14x14.wav',
             'naturalmothcalls/gl005_11x11.wav',
             'naturalmothcalls/gl116_05x05.wav',
             'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
             'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/melese_12x12_01_PK1297.wav',
             'naturalmothcalls/neritos_cotes_10x10.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
             'naturalmothcalls/syntrichura_12x12.wav']

    call_stats = []
    file_pathname = '/media/brehm/Data/MasterMoth/stimuli_backup/'

    for k in range(len(stims)):
        file_name = file_pathname + stims[k][0:-4] + '/call_stats.xls'
        try:
            with open(file_name, newline='') as f:
                df = pd.read_excel(file_name)
                call_stats.append(df.values[0])
            with open('/media/brehm/Data/MasterMoth/outfile.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ')
                a = list(call_stats[k])
                a.insert(0, stims[k][17:-10])
                writer.writerow(a)
        except:
            print(stims[k] + ' not found')
            with open('/media/brehm/Data/MasterMoth/outfile.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ')
                a = list([np.nan] * 18)
                a.insert(0, stims[k][17:-10])
                writer.writerow(a)
    embed()
    exit()

if CALLSFROMMATLAB:
    pd_a = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/pd_a.mat')['pd_a_py'][0]
    pd_p = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/pd_p.mat')['pd_p_py'][0]

    ipi_a = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/ipi_a.mat')['ipi_a_py'][0]
    ipi_p = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/ipi_p.mat')['ipi_p_py'][0]

    freq_a = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/freq_a.mat')['freq_a_py'][0]
    freq_p = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/freq_p.mat')['freq_p_py'][0]

    pnr_a = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/pnr_a.mat')['pnr_a_py'][0]
    pnr_p = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/pnr_p.mat')['pnr_p_py'][0]

    ITI = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/ITI.mat')['ITI_py'][0]
    call_dur = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/calldur.mat')['call_dur_py'][0]

    # Bar Plot

    mf.plot_settings()
    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=26)
    fig = plt.figure(figsize=(5.9, 1.9))
    ax1 = plt.subplot(grid[0, 0:10])
    ax2 = plt.subplot(grid[0, 15:25])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    width = 0.35
    ind = np.arange(0, 20, 1)
    ax1.bar(ind, ITI, width, label='ITI', color='black')
    ax1.bar(ind + width, call_dur, width, label='Call dur.', color='grey')
    ax1.set_xticks(ind + width / 2)
    ax1.legend(frameon=False, loc=1)

    ax2.bar(ind, pnr_a, width, label='Active', color='black')
    ax2.bar(ind + width, pnr_p, width, label='Passive', color='grey')
    ax2.set_xticks(ind + width / 2)
    ax2.legend(frameon=False, loc=1)

    xlbls = [''] * 20
    xlbls[0], xlbls[4], xlbls[9],xlbls[14], xlbls[19] = 1, 5, 10, 15, 20
    ax1.set_xticklabels(xlbls)
    ax2.set_xticklabels(xlbls)

    fig.text(0.5, 0.05, 'Call number', ha='center', fontdict=None)
    ax1.set_ylabel('Duration [ms]')
    ax2.set_ylabel('Pulse count')

    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')

    ax1.set_yticks(np.arange(0, 201, 50))
    ax1.set_ylim(0, 200)
    ax2.set_yticks(np.arange(0, 31, 5))
    ax2.set_ylim(0, 30)

    sns.despine()
    figname = '/media/brehm/Data/MasterMoth/CallStats/CallStats_bar.pdf'
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(figname)
    plt.close(fig)
    exit()



    # BOXPLOT
    mf.plot_settings()
    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=23, ncols=41)
    fig = plt.figure(figsize=(5.9, 3.9))
    ax1 = plt.subplot(grid[0:10, 0:10])
    ax2 = plt.subplot(grid[0:10, 15:25])
    ax3 = plt.subplot(grid[0:10, 30:40])
    ax4 = plt.subplot(grid[12:22, 0:10])
    ax5 = plt.subplot(grid[12:22, 15:25])
    ax6 = plt.subplot(grid[12:22, 30:40])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.90
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    ax1.boxplot(pd_a, showcaps=False, showfliers=False)
    for k in range(len(freq_a)):
        ax1.plot(np.zeros(len(pd_a[k]))+k+1, pd_a[k], 'ko', markersize=1)

    ax2.boxplot(ipi_a, showcaps=False, showfliers=False)
    for k in range(len(ipi_a)):
        ax2.plot(np.zeros(len(ipi_a[k])) + k + 1, ipi_a[k], 'ko', markersize=1)

    ax3.boxplot(freq_a, showcaps=False, showfliers=False)
    for k in range(len(freq_a)):
        ax3.plot(np.zeros(len(freq_a[k])) + k + 1, freq_a[k], 'ko', markersize=1)

    ax4.boxplot(pd_p, showcaps=False, showfliers=False)
    for k in range(len(freq_p)):
        ax4.plot(np.zeros(len(pd_p[k])) + k + 1, pd_p[k], 'ko', markersize=1)

    ax5.boxplot(ipi_p, showcaps=False, showfliers=False)
    for k in range(len(ipi_p)):
        ax5.plot(np.zeros(len(ipi_p[k])) + k + 1, ipi_p[k], 'ko', markersize=1)

    ax6.boxplot(freq_p, showcaps=False, showfliers=False)
    for k in range(len(freq_p)):
        ax6.plot(np.zeros(len(freq_p[k])) + k + 1, freq_p[k], 'ko', markersize=1)

    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')
    ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
             color='black')
    ax5.text(label_x_pos, label_y_pos, subfig_caps_labels[4], transform=ax5.transAxes, size=subfig_caps,
             color='black')
    ax6.text(label_x_pos, label_y_pos, subfig_caps_labels[5], transform=ax6.transAxes, size=subfig_caps,
             color='black')

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    xlbls = [''] * 20
    xlbls[0], xlbls[4], xlbls[9],xlbls[14], xlbls[19] = 1, 5, 10, 15, 20
    ax4.set_xticklabels(xlbls)
    ax5.set_xticklabels(xlbls)
    ax6.set_xticklabels(xlbls)

    ax1.set_ylim(0, 0.6)
    ax4.set_ylim(0, 0.6)
    ax2.set_ylim(0, 20)
    ax5.set_ylim(0, 20)
    ax3.set_ylim(0, 80)
    ax6.set_ylim(0, 80)

    fig.text(0.5, 0.05, 'Call number', ha='center', fontdict=None)
    fig.text(0.925, 0.80, 'Active pulses', ha='center', fontdict=None, rotation=-90, color='red')
    fig.text(0.925, 0.40, 'Passive pulses', ha='center', fontdict=None, rotation=-90, color='blue')

    ax1.set_ylabel('Pulse duration [ms]')
    ax2.set_ylabel('IPI [ms]')
    ax3.set_ylabel('Frequency [kHz]')
    ax4.set_ylabel('Pulse duration [ms]')
    ax5.set_ylabel('IPI [ms]')
    ax6.set_ylabel('Frequency [kHz]')
    sns.despine()

    figname = '/media/brehm/Data/MasterMoth/CallStats/CallStats.pdf'
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(figname)
    plt.close(fig)

if CALLSERIESFROMMATLAB:
    # Call Series
    # samples = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
    # Single Calls
    samples = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
    # samples = sio.loadmat('/media/brehm/Data/MasterMoth/CallStats/samples.mat')['samples'][0]

    # Raster Plot
    mf.plot_settings()
    tts = [0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 1.5]
    # tts = [0.01, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]

    for i in range(len(tts)):
        time_limit = tts[i]
        step = 0.1
        w = 0
        fig, ax = plt.subplots(figsize=(5.9, 3.9))
        # fig = plt.figure(figsize=(5.9, 1.9))
        ending = [[]] * len(samples)
        for k in range(len(samples)):
            idx = samples[k] <= time_limit
            count = np.sum(idx)

            ax.plot([0, time_limit], [w, w], color='0.8')
            ax.plot(samples[k], np.zeros(len(samples[k]))+w, 'k|')
            ax.plot(np.max(samples[k][idx]), w, 'r|', markersize=3)
            # ax.plot(np.max(samples[k][idx]), len(samples)*step+step, 'g|', markersize=3)
            try:
                ax.text(time_limit, w-0.01, str(count) + ' ; ' + str(int(np.round(np.mean(np.diff(np.sort(samples[k][idx])))*1000))) + ' ms', size=5)
            except:
                ax.text(time_limit, w-0.01, str(count) + ' ; -', size=5)
            if i == len(tts)-1:
                ending[k] = np.max(samples[k][idx])
            w += step

        # if i == len(tts)-1:
        #     # plot the cumulative histogram
        #     a= plt.axes([.55, .7, .32, .175], facecolor='w')
        #     # a = fig.add_subplot(40, 1, 1)
        #     n_bins = int(tts[i] / 0.005)
        #     n, bins, patches = a.hist(ending, n_bins, normed=0, histtype='step', cumulative=True, label='CumHist')
        #     a.set_xlim(0, time_limit)
        #     plt.xticks(fontsize=6)
        #     plt.yticks(fontsize=6)
        #     # a.set_xticks([])
        #     a.set_yticks(np.arange(0, 20.1, 5))
        #     a.set_xlabel('Time [s]', size=6)
        #     a.set_ylabel('Count of finished calls', size=6)
        #     # a.patch.set_facecolor('white')

        ax.set_yticks(np.arange(0, len(samples)*step, step))
        ax.set_yticklabels(np.arange(0, len(samples), 1))
        ax.set_ylabel('Call number')
        ax.set_xlabel('Time [s]')
        sns.despine()
        ax.set_xlim(0, time_limit)

        fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
        figname = '/media/brehm/Data/MasterMoth/CallStats/CallSeriesRasterPlot_' + str(int(time_limit*1000)) +'.pdf'
        # figname = '/media/brehm/Data/MasterMoth/CallStats/SingleCallsRasterPlot_' + str(int(time_limit*1000)) +'.pdf'
        fig.savefig(figname)
        plt.close()
        np.save('/media/brehm/Data/MasterMoth/CallStats/' + 'endings_CallSeries.npy', ending)
        # np.save('/media/brehm/Data/MasterMoth/CallStats/' + 'endings_SingleCalls.npy', ending)

if CUMHIST:
    endings_callseries = np.load('/media/brehm/Data/MasterMoth/CallStats/' + 'endings_CallSeries.npy')
    endings_singlecalls = np.load('/media/brehm/Data/MasterMoth/CallStats/' + 'endings_SingleCalls.npy')

    # Plot
    mf.plot_settings()
    fig = plt.figure(figsize=(5.9, 2.9))
    grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=46)
    ax1 = plt.subplot(grid[0, 0:20])
    ax2 = plt.subplot(grid[0, 25:45])

    n_bins = int(0.2 / 0.005)
    n, bins, patches = ax1.hist(endings_singlecalls, n_bins, normed=0, histtype='stepfilled', cumulative=True, label='CumHist', color='0.75')
    n, bins, patches = ax1.hist(endings_singlecalls, n_bins, normed=0, histtype='step', cumulative=True,
                                label='CumHist', color='k')

    n_bins = int(2 / 0.005)
    n, bins, patches = ax2.hist(endings_callseries, n_bins, normed=0, histtype='stepfilled', cumulative=True,
                                label='CumHist', color='0.75')
    n, bins, patches = ax2.hist(endings_callseries, n_bins, normed=0, histtype='step', cumulative=True,
                                label='CumHist', color='k')

    ax1.set_ylabel('Count of finished calls')
    fig.text(0.5, 0.035, 'Time [s]', ha='center', fontdict=None)
    sns.despine()

    ax1.set_yticks(np.arange(0, 20.1, 5))
    ax2.set_yticks(np.arange(0, 20.1, 5))
    ax1.set_ylim(0, 20)
    ax2.set_ylim(0, 20)

    ax1.set_xticks(np.arange(0, 0.21, 0.05))
    ax2.set_xticks(np.arange(0, 2.1, 0.5))
    ax1.set_xlim(0, 0.2)
    ax2.set_xlim(0, 2)

    ax1.grid(color='0.3', linestyle='--', linewidth=.5)
    ax2.grid(color='0.3', linestyle='--', linewidth=.5)

    # Subfig caps
    subfig_caps = 12
    label_x_pos = -0.025
    label_y_pos = 0.95
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    ax1.text(label_x_pos - 0.2, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos - 0.2, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')

    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)
    figname = '/media/brehm/Data/MasterMoth/CallStats/CumHists.pdf'
    fig.savefig(figname)
    plt.close()

if PLOT_VR_SPIKEMATCHING:
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]
    # Get all data
    s_types = ['moth_series_selected', 'moth_series_selected_extended', 'moth_single_selected',
               'moth_single_selected_extended']
    data = {}
    for k in range(len(s_types)):
        data.update({s_types[k]: np.load(p + 'VanRossum_matches_' + s_types[k] + '.npy').item()})

    # Plot
    mf.plot_settings()
    ax = [[]] * 13
    grid = matplotlib.gridspec.GridSpec(nrows=35, ncols=57)
    fig = plt.figure(figsize=(5.9, 3.9))
    ax[0] = plt.subplot(grid[0:10, 0:10])
    ax[1] = plt.subplot(grid[0:10, 14:24])

    ax[2] = plt.subplot(grid[0:10, 30:40])
    ax[3] = plt.subplot(grid[0:10, 44:54])

    ax[4] = plt.subplot(grid[12:22, 0:10])
    ax[5] = plt.subplot(grid[12:22, 14:24])

    ax[6] = plt.subplot(grid[12:22, 30:40])
    ax[7] = plt.subplot(grid[12:22, 44:54])

    ax[8] = plt.subplot(grid[24:34, 0:10])
    ax[9] = plt.subplot(grid[24:34, 14:24])

    ax[10] = plt.subplot(grid[24:34, 30:40])
    ax[11] = plt.subplot(grid[24:34, 44:54])
    # Colorbar ax
    ax[12] = plt.subplot(grid[1:33, 55:56])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = 0.05
    label_y_pos = 0.82
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

    grid_color = '0.75'
    grid_linewidth = 0.5

    Xticks = [[]] * 4
    Xticks[0] = np.arange(0, 16, 5)
    Xticks[1] = np.arange(0, 16, 5)
    Xticks[2] = np.arange(0, 19, 5)
    Xticks[3] = np.arange(0, 19, 5)
    Xticks = Xticks * 3

    DURS = [20, 20, 40, 40] * 3
    TAUS = [0, 0, 0, 0, 9, 9, 9, 9, 34, 34, 34, 34]
    tau_text = [r'$\tau$=1 ms', r'$\tau$=1 ms', r'$\tau$=1 ms', r'$\tau$=1 ms', r'$\tau$=10 ms', r'$\tau$=10 ms',
                r'$\tau$=10 ms', r'$\tau$=10 ms', r'$\tau$=100 ms', r'$\tau$=100 ms', r'$\tau$=100 ms', r'$\tau$=100 ms']
    s_types = s_types * 3
    # ColorMeshPlot
    for i in range(len(ax)-1):
        # X, Y = np.meshgrid(XX[i], XX[i])
        dd = data[s_types[i]][taus[TAUS[i]]][DURS[i]]
        im = ax[i].pcolormesh(dd, cmap='jet', vmin=0, vmax=20, shading='flat', rasterized=True)

        # ax[i].set_yscale('log')
        ax[i].set_xticks(Xticks[i])
        ax[i].set_yticks(Xticks[i])
        ax[i].set_aspect('equal')
        # Subplot caps
        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[i], transform=ax[i].transAxes, size=subfig_caps,
                   color='white')

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
    cb1 = matplotlib.colorbar.ColorbarBase(ax[12], cmap='jet', norm=norm)

    ax[4].set_ylabel('Matched call')

    fig.text(0.5, 0.025, 'Original call', ha='center', fontdict=None)

    sz_text = 12
    fig.text(0.19, 0.9, 'Call series', ha='center', fontdict=None, size=sz_text)
    fig.text(0.38, 0.9, 'Cs extended', ha='center', fontdict=None, size=sz_text)
    fig.text(0.6, 0.9, 'Single calls', ha='center', fontdict=None, size=sz_text)
    fig.text(0.79, 0.9, 'Sc extended', ha='center', fontdict=None, size=sz_text)

    fig.text(0.02, 0.8, r'$\tau$=1 ms', ha='center', fontdict=None, rotation=90, size=sz_text)
    fig.text(0.02, 0.54, r'$\tau$=10 ms', ha='center', fontdict=None, rotation=90, size=sz_text)
    fig.text(0.02, 0.28, r'$\tau$=100 ms', ha='center', fontdict=None, rotation=90, size=sz_text)

    fig.text(0.96, 0.55, 'Spike trains', ha='center', fontdict=None, rotation=-90)

    # fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(path_names[2] + 'final/VanRossum_SpikeTrainMatching.pdf')
    plt.close(fig)
    print('Van Rossum Spike Train Matching Plot saved')

if PLOT_DISTANCES_CORRECT:
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]

    plot_distances = True
    plot_correct = False
    # Get all data
    s_types = ['moth_series_selected', 'moth_series_selected_extended', 'moth_single_selected',
               'moth_single_selected_extended']
    data_correct = {}
    data_correct_random = {}
    data_ratios = {'ISI': {}, 'SYNC': {}, 'DUR': {}, 'COUNT': {}}
    n = [17, 17, 20, 20]
    for k in range(len(s_types)):
        data_correct.update({s_types[k]: np.load(path_names[1] + 'distances_correct_' + s_types[k] + '.npy')})
        data_correct_random.update({s_types[k]: np.load(path_names[1] + 'distances_rand_correct_' + s_types[k] + '.npy')})

        data_ratios['ISI'].update({s_types[k]: np.load(path_names[1] + 'ISI_Ratios_' + s_types[k] + '.npy')})
        data_ratios['SYNC'].update({s_types[k]: np.load(path_names[1] + 'SYNC_Ratios_' + s_types[k] + '.npy')})
        # data_ratios['DUR'].update({s_types[k]: np.load(path_names[1] + 'DUR_Ratios_' + s_types[k] + '.npy')})
        # data_ratios['COUNT'].update({s_types[k]: np.load(path_names[1] + 'COUNT_Ratios_' + s_types[k] + '.npy')})

        # Normalize
        a = np.load(path_names[1] + 'DUR_Ratios_' + s_types[k] + '.npy')
        norm_max = np.max(a[:, 2])
        a[:, 0] = a[:, 0] / norm_max
        a[:, 1] = a[:, 1] / norm_max
        a[:, 4] = a[:, 4] / norm_max
        a[:, 2] = a[:, 2] / norm_max
        data_ratios['DUR'].update({s_types[k]: a})

        b = np.load(path_names[1] + 'COUNT_Ratios_' + s_types[k] + '.npy')
        norm_max = np.max(b[:, 2])
        b[:, 0] = b[:, 0] / norm_max
        b[:, 1] = b[:, 1] / norm_max
        b[:, 4] = b[:, 4] / norm_max
        b[:, 2] = b[:, 2] / norm_max
        data_ratios['COUNT'].update({s_types[k]: b})

    if plot_correct:
        # Plot Correct vs. Duration
        mf.plot_settings()
        ax = [[]] * 4
        grid = matplotlib.gridspec.GridSpec(nrows=25, ncols=25)
        fig = plt.figure(figsize=(5.9, 4.9))
        ax[0] = plt.subplot(grid[0:10, 0:10])
        ax[1] = plt.subplot(grid[0:10, 14:24])

        ax[2] = plt.subplot(grid[14:24, 0:10])
        ax[3] = plt.subplot(grid[14:24, 14:24])

        du1 = list(np.arange(0, 255, 5))
        du1[0] = 1
        du2 = list(np.arange(0, 2550, 50))
        du2[0] = 10
        duration = [du2, du2, du1, du1]
        marks = ['', 'o', 'v', 's', '']
        cc = ['0', 'orangered', 'navy', 'teal', '0']
        styles = [':', '-', '-.', '-', '--']
        # Subplotfig caps
        subfig_caps = 12
        label_x_pos = -0.3
        label_y_pos = 1.08
        subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        labs = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR']
        for i in range(len(ax)):
            ax[i].plot(duration[i], data_correct_random[s_types[i]][:, 0], marker='', color='0.9', linestyle='-', linewidth=3)
            for k in range(5):
                ax[i].plot(duration[i], data_correct[s_types[i]][:, k], marker=marks[k], color=cc[k], linestyle=styles[k],
                           label=labs[k], markersize=0.5)
                ax[i].set_ylim(0, 1)
                ax[i].set_yticks(np.arange(0, 1.1, 0.25))
            ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels[i], transform=ax[i].transAxes, size=subfig_caps,
                         color='black')
            # ax[i].grid(color='0.3', linestyle='-', linewidth=.5)
        ax[3].legend(frameon=False)

        ax[0].set_xticks(np.arange(0, 2550, 500))
        ax[0].set_xlim(-100, 2500)

        ax[1].set_xticks(np.arange(0, 2550, 500))
        ax[1].set_xlim(-100, 2500)

        ax[2].set_xticks(np.arange(0, 255, 50))
        ax[2].set_xlim(-10, 250)

        ax[3].set_xticks(np.arange(0, 255, 50))
        ax[3].set_xlim(-10, 250)

        ax[0].set_ylabel('Correct')
        ax[2].set_ylabel('Correct')
        fig.text(0.5, 0.03, 'Spike train duration [ms]', ha='center', fontdict=None)
        # fig.text(0.02, 0.5, 'Correct', ha='center', fontdict=None, rotation=90)

        sns.despine()
        fig.savefig(path_names[2] + 'final/Distances_Correct_TEST.pdf')
        plt.close(fig)
        print('Distances Correct plot saved')

    # Plot Distances and Ratios
    if plot_distances:
        mf.plot_settings()
        ax = [[]] * 24
        grid = matplotlib.gridspec.GridSpec(nrows=81, ncols=59)
        fig = plt.figure(figsize=(5.9, 6.9))
        ax[0] = plt.subplot(grid[0:10, 0:10])
        ax[1] = plt.subplot(grid[0:10, 14:24])

        ax[2] = plt.subplot(grid[0:10, 34:44])
        ax[3] = plt.subplot(grid[0:10, 48:58])

        ax[4] = plt.subplot(grid[14:24, 0:10])
        ax[5] = plt.subplot(grid[14:24, 14:24])

        ax[6] = plt.subplot(grid[14:24, 34:44])
        ax[7] = plt.subplot(grid[14:24, 48:58])

        ax[8] = plt.subplot(grid[28:38, 0:10])
        ax[9] = plt.subplot(grid[28:38, 14:24])

        ax[10] = plt.subplot(grid[28:38, 34:44])
        ax[11] = plt.subplot(grid[28:38, 48:58])

        ax[12] = plt.subplot(grid[42:52, 0:10])
        ax[13] = plt.subplot(grid[42:52, 14:24])

        ax[14] = plt.subplot(grid[42:52, 34:44])
        ax[15] = plt.subplot(grid[42:52, 48:58])

        ax[16] = plt.subplot(grid[56:66, 0:10])
        ax[17] = plt.subplot(grid[56:66, 14:24])

        ax[18] = plt.subplot(grid[56:66, 34:44])
        ax[19] = plt.subplot(grid[56:66, 48:58])

        ax[20] = plt.subplot(grid[70:80, 0:10])
        ax[21] = plt.subplot(grid[70:80, 14:24])

        ax[22] = plt.subplot(grid[70:80, 34:44])
        ax[23] = plt.subplot(grid[70:80, 48:58])

        du1 = list(np.arange(0, 255, 5))
        du1[0] = 1
        du2 = list(np.arange(0, 2550, 50))
        du2[0] = 10
        duration = [du2, du2, du1, du1]
        marks = ['', 'o', 'v', 's', '']
        cc = ['0', 'orangered', 'navy', 'teal', '0']
        styles = [':', '-', '-', '-', '--']

        # Subplotfig caps
        subfig_caps = 12
        label_x_pos = -0.5
        label_y_pos = 1.1
        subfig_caps_labels1 = ['a', 'c', 'e',  'g' ,'i', 'k' , 'm']
        subfig_caps_labels2 = ['b', 'd', 'f', 'h' ,'j' ,'l', 'n']

        a = np.arange(0, 21, 4)
        b = np.arange(1, 22, 4)
        c = np.arange(2, 23, 4)
        d = np.arange(3, 24, 4)

        for i in range(len(a)):
            ax[a[i]].set_xticks(np.arange(0, 2550, 1000))
            ax[a[i]].set_xticklabels([])
            ax[b[i]].set_xticks(np.arange(0, 2550, 1000))
            ax[b[i]].set_xticklabels([])
            ax[b[i]].set_yticklabels([])
            ax[c[i]].set_xticks(np.arange(0, 255, 100))
            ax[c[i]].set_xticklabels([])
            ax[d[i]].set_xticks(np.arange(0, 255, 100))
            ax[d[i]].set_xticklabels([])
            ax[d[i]].set_yticklabels([])
            ax[a[i]].text(label_x_pos, label_y_pos, subfig_caps_labels1[i], transform=ax[a[i]].transAxes, size=subfig_caps,
                       color='black')
            ax[c[i]].text(label_x_pos, label_y_pos, subfig_caps_labels2[i], transform=ax[c[i]].transAxes, size=subfig_caps,
                          color='black')

        ax[20].set_xticklabels([0, 1, 2])
        ax[21].set_xticklabels([0, 1, 2])
        ax[22].set_xticklabels([0, 0.1, 0.2])
        ax[23].set_xticklabels([0, 0.1, 0.2])

        ax[1].set_yticklabels([])
        ax[3].set_yticklabels([])

        duration = duration * 4
        s_types = s_types * 4
        profiles = ['ISI', 'ISI', 'ISI', 'ISI', 'SYNC', 'SYNC', 'SYNC', 'SYNC', 'DUR', 'DUR', 'DUR', 'DUR', 'COUNT',
                    'COUNT', 'COUNT', 'COUNT']
        n = [17, 17, 20, 20] * 4
        # Plot Distances
        for i in range(0, 16):
            # if profiles[i] == 'DUR':
            #     data_ratios[profiles[i]][s_types[i]][:, 0] = data_ratios[profiles[i]][s_types[i]][:, 0] * 1000
            #     data_ratios[profiles[i]][s_types[i]][:, 1] = data_ratios[profiles[i]][s_types[i]][:, 1] * 1000
            #     data_ratios[profiles[i]][s_types[i]][:, 2] = data_ratios[profiles[i]][s_types[i]][:, 2] * 1000
            sem = data_ratios[profiles[i]][s_types[i]][:, 1] / np.sqrt(n[i])
            ax[i].fill_between(duration[i], data_ratios[profiles[i]][s_types[i]][:, 0] - sem,
                             data_ratios[profiles[i]][s_types[i]][:, 0] + sem, facecolors='k', alpha=0.25)
            ax[i].plot(duration[i], data_ratios[profiles[i]][s_types[i]][:, 0], 'k')
            ax[i].plot(duration[i], data_ratios[profiles[i]][s_types[i]][:, 2], 'b')
            ax[i].set_ylim(0, 1)
            ax[i].set_yticks([0, 0.5, 1])

        # ax[8].set_ylim(0, 200)
        # ax[8].set_yticks(np.arange(0, 200+5, 100))
        # ax[9].set_ylim(0, 200)
        # ax[9].set_yticks(np.arange(0, 200+5, 100))
        #
        # ax[10].set_ylim(0, 100)
        # ax[10].set_yticks(np.arange(0, 100 + 2, 50))
        # ax[11].set_ylim(0, 100)
        # ax[11].set_yticks(np.arange(0, 100 + 2, 50))
        #
        # ax[12].set_ylim(0, 40)
        # ax[12].set_yticks(np.arange(0, 40 + 1, 20))
        # ax[13].set_ylim(0, 40)
        # ax[13].set_yticks(np.arange(0, 40 + 1, 20))
        #
        # ax[14].set_ylim(0, 20)
        # ax[14].set_yticks(np.arange(0, 20 + 1, 10))
        # ax[15].set_ylim(0, 20)
        # ax[15].set_yticks(np.arange(0, 20 + 1, 10))

        # Plot Diffs
        # mode: ratio = 3, diff = 4
        mode = 4
        cc = ['0', 'orangered', 'navy', 'teal', '0']
        for i in range(16, 20):
            k = i-16
            ax[i].plot(duration[k], data_ratios['ISI'][s_types[k]][:, mode], marker='', color='orangered', linestyle='-')
            ax[i].plot(duration[k], -data_ratios['SYNC'][s_types[k]][:, mode], marker='', color='teal', linestyle='-')
            ax[i].plot(duration[k], data_ratios['DUR'][s_types[k]][:, mode], marker='', color='0', linestyle='--')
            ax[i].plot(duration[k], data_ratios['COUNT'][s_types[k]][:, mode], marker='', color='0', linestyle=':')
            ax[i].set_ylim(-0.1, 0.4)
            ax[i].set_yticks([0, 0.2, 0.4])

        # Plot Ratios
        # mode: ratio = 3, diff = 4
        mode = 3
        cc = ['0', 'orangered', 'navy', 'teal', '0']
        for i in range(20, 24):
            k = i - 20
            ax[i].plot(duration[k], data_ratios['ISI'][s_types[k]][:, mode], marker='', color='orangered',
                       linestyle='-')
            ax[i].plot(duration[k], 1/data_ratios['SYNC'][s_types[k]][:, mode], marker='', color='teal',
                       linestyle='-')
            ax[i].plot(duration[k], data_ratios['DUR'][s_types[k]][:, mode], marker='', color='0', linestyle='--')
            ax[i].plot(duration[k], data_ratios['COUNT'][s_types[k]][:, mode], marker='', color='0', linestyle=':')
            ax[i].set_ylim(0.9, 3)
            ax[i].set_yticks([1, 2, 3])

        # fig.text(0.05, 0.85, 'Correct', ha='center', fontdict=None, rotation=90)
        fig.text(0.054, 0.835, 'ISI', ha='center', fontdict=None, rotation=90, va='center')
        fig.text(0.054, 0.7, 'SYNC', ha='center', fontdict=None, rotation=90, va='center')
        fig.text(0.054, 0.57, 'Norm. \nDUR', ha='center', fontdict=None, rotation=90, va='center')
        fig.text(0.054, 0.44, 'Norm. \nCOUNT', ha='center', fontdict=None, rotation=90, va='center')
        fig.text(0.054, 0.3, 'Interspace', ha='center', fontdict=None, rotation=90, va='center')
        fig.text(0.054, 0.17, 'Ratio', ha='center', fontdict=None, rotation=90, va='center')

        fig.text(0.5, 0.03, 'Spike train duration [s]', ha='center', fontdict=None)

        sz_text = 12
        fig.text(0.18, 0.93, 'Call series', ha='center', fontdict=None, size=sz_text)
        fig.text(0.38, 0.93, 'Cs extended', ha='center', fontdict=None, size=sz_text)
        fig.text(0.64, 0.93, 'Single calls', ha='center', fontdict=None, size=sz_text)
        fig.text(0.825, 0.93, 'Sc extended', ha='center', fontdict=None, size=sz_text)
        sns.despine()
        fig.savefig(path_names[2] + 'final/Distances_DiffsAndRatio.pdf')
        plt.close(fig)
        print('Distances Ratios plot saved')

if POISSON_TAU_CORRECT:
    data_name = '2018-02-16-aa'
    path_names = mf.get_directories(data_name=data_name)
    print(data_name)
    p = path_names[1]

    # Get all data
    data = {}
    data.update({'vr': np.load(p + 'VanRossum_correct_' + 'poisson' + '.npy')})
    data.update({'distances_same': np.load(p + 'distances_correct_poisson_same' + '.npy')})
    data.update({'distances_diff': np.load(p + 'distances_correct_poisson_diff' + '.npy')})
    data.update({'distances_2diff': np.load(p + 'distances_correct_poisson_2diff' + '.npy')})
    data.update({'random': np.load(p + 'distances_rand_correct_poisson' + '.npy')})

    # Plot
    mf.plot_settings()
    ax = [[]] * 5
    grid = matplotlib.gridspec.GridSpec(nrows=11, ncols=53)
    fig = plt.figure(figsize=(5.9, 2.4))
    ax[0] = plt.subplot(grid[0:10, 0:20])
    ax[1] = plt.subplot(grid[0:10, 21:22])
    ax[2] = plt.subplot(grid[0:10, 32:52])

    # Image Grid
    x_series = np.arange(0, 2550, 50)
    x_series[0] = 10
    x_series = x_series / 1000
    y = taus

    # Subplot caps
    subfig_caps = 12
    label_x_pos = -0.25
    label_y_pos = 1.05
    subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    grid_color = '0.75'
    grid_linewidth = 0.5

    Xticks = np.arange(0, 2550, 500)/1000

    # Tau vs. Duration
    X, Y = np.meshgrid(x_series, y)
    im = ax[0].pcolormesh(X, Y, data['vr'].T, cmap='jet', vmin=0, vmax=1, shading='flat', rasterized=True)
    ax[0].set_yscale('log')
    ax[0].set_xticks(Xticks)
    ax[0].text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax[0].transAxes, size=subfig_caps,
               color='black')
    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax[1], cmap='jet', norm=norm)
    # cb1.set_label('Correct')

    # Correct vs Duration
    duration = np.arange(0, 2550, 50)
    duration[0] = 10
    marks = ['', 'o', 'v', 's', '']
    cc = ['0', 'orangered', 'navy', 'teal', '0']
    styles = [':', '-', '-.', '-', '--']
    labs = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR']
    for i in range(5):
        ax[2].plot(duration/1000, data['distances_same'][:, i], marker='', color=cc[i], linestyle=styles[i], label=labs[i])
        ax[2].plot(duration/1000, data['distances_diff'][:, i], marker='', color=cc[i], linestyle=styles[i], label=labs[i], lw=1.5)
        ax[2].plot(duration/1000, data['distances_2diff'][:, i], marker='', color=cc[i], linestyle=styles[i], label=labs[i], lw=2.5)

    ax[2].text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax[2].transAxes, size=subfig_caps,
               color='black')
    ax[2].legend(frameon=False, loc=0)
    ax[2].set_xticks(Xticks)
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks(np.arange(0, 1.1, 0.2))
    sns.despine(ax=ax[2])
    fig.text(0.5, 0.025, 'Spike train duration [s]', ha='center', fontdict=None)
    fig.text(0.025, 0.55, r'$\tau$ [ms]', ha='center', fontdict=None, rotation=90)
    fig.text(0.51, 0.55, 'Correct', ha='center', fontdict=None, rotation=-90)

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.1, hspace=0.1)
    fig.savefig(path_names[2] + 'final/Poisson_TauVsDur_Correct.pdf')
    plt.close(fig)
    print('Poisson Plot saved')

if TEST:
    data = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    mf.plot_settings()
    plt.plot(data, data, 'rs--')
    plt.show()
    embed()
print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))


