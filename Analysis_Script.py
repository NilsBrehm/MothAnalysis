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

datasets = ['2018-02-20-aa']

FIFIELD = False
INTERVAL_MAS = True
Bootstrapping = False
INTERVAL_REC = False
GAP = False
SOUND = False

EPULSES = False
ISI = False
VANROSSUM = False
PULSE_TRAIN_ISI = False
PULSE_TRAIN_VANROSSUM = False

FI_OVERANIMALS = False
PLOT_CORRECT = False

SELECT = True

# **********************************************************************************************************************
# Settings for Spike Detection =========================================================================================
th_factor = 3
mph_percent = 2
bin_size = 0.005
# If true show plots (list: 0: spike detection, 1: overview, 2: vector strength)
show = False

# Settings for Call Analysis ===========================================================================================
# General Settings
stim_type = 'moth_single'
duration = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200] if stim_type == 'moth_single' else False
duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500] if stim_type == 'moth_series' else False

# ISI and co
# profs = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR', 'VanRossum']
profs = ['COUNT', 'ISI', 'DUR']
profs_plot_correct = ['COUNT', 'ISI', 'SPIKE', 'SYNC', 'DUR', 'VanRossum']

# Bootstrapping
nsamples = 10

# VanRossum
whole_train = True
method = 'exp'
dt_factor = 100
taus = [1, 5, 10, 20, 30, 50]
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
    datasets = sorted(datasets)

# Get relative paths ===================================================================================================
# RECT: good recs: all Estigmene
# datasets = ['2017-11-27-aa', '2017-11-29-aa', '2017-12-04-aa', '2018-02-16-aa', '2017-12-05-ab']

print('data set count: ' + str(len(datasets)))

# data_name = datasets[4]
# print(data_name)
#
# #  path_names = [data_name, data_files_path, figs_path, nix_path]
# path_names = mf.get_directories(data_name=data_name)


if GAP:
    # dat = datasets[5]
    # dat = datasets[-2]
    protocol_name = 'Gap'
    spike_detection = False
    data_name = datasets[-2]
    print(data_name)
    path_names = mf.get_directories(data_name=data_name)

    # tag_list = np.load(path_names[1] + 'Gap_tag_list.npy')
    if spike_detection:
        show_detection = False
        mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor, filter_on=True,
                           window=None, mph_percent=mph_percent)
    mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=False, show=True, save_data=False, old=False)

# Rect Intervals
if INTERVAL_REC:
    protocol_name = 'PulseIntervalsRect'
    spike_detection = False
    for dat in range(len(datasets)):
        print(str(dat) + ' of ' + str(len(datasets)))
        data_name = datasets[dat]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        if spike_detection:
            show_detection = False
            mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                               filter_on=True,
                               window=None, mph_percent=mph_percent)
        mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=True, save_data=False, old=False)
        exit()

    # mf.plot_cohen(protocol_name, datasets, save_fig=True)

# Analyse Intervals MothASongs data stored on HDD
if INTERVAL_MAS:
    old = True
    protocol_name = 'intervals_mas'
    spike_detection = False
    show_detection = False
    data_name = datasets[-2]
    # old 0:9
    print(data_name)
    if old:
        print('OLD MAS Protocol!')
    path_names = mf.get_directories(data_name=data_name)

    if spike_detection:
        mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                           filter_on=True, window=None, mph_percent=mph_percent)

    mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=[True, True], save_data=False, old=old)

    # OLD:
    # mf.moth_intervals_spike_detection(path_names, window=None, th_factor=th_factor, mph_percent=mph_percent,
    #                                   filter_on=True, save_data=False, show=show, bin_size=bin_size)
    # mf.moth_intervals_analysis(datasets[0])

# Analyse FIField data stored on HDD
if FIFIELD:
    save_plot = True
    plot_fi_field = True
    single_fi = True
    data = path_names[0]
    p = path_names[1]
    th = 6
    spike_count, fi_field, fsl = mf.fifield_analysis2(path_names, th, plot_fi=False)
    # freqs = np.zeros(len(spike_count))
    freqs = [[]] * len(spike_count)
    i = 0
    for key in spike_count:
        freqs[i] = int(key)
        i += 1
    freqs = sorted(freqs)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    plt.rc('font', **font)

    if plot_fi_field:
        plt.plot(fi_field[:, 0], fi_field[:, 1], 'ko-')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('dB SPL at Threshold (' + str(th) + ' spikes)')
        plt.ylim(0, 90)
        plt.yticks(np.arange(0, 90, 20))
        plt.xlim(10, 110)
        plt.xticks(np.arange(20, 110, 10))
        if save_plot:
            # Save Plot to HDD
            figname = p + 'fi_field.png'
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print('Plot saved for ' + data)
        else:
            plt.show()

    # Plot single FI Curve
    if single_fi:
        ff = 55
        fig, ax1 = plt.subplots()
        color = 'k'
        ax1.set_xlabel('Sound Pressure Level [dB SPl]')
        ax1.set_ylabel('Spike Count per Stimulus', color=color)
        ax1.errorbar(spike_count[ff][:, 0], spike_count[ff][:, 1], yerr=spike_count[ff][:, 2],
                     marker='o', color='k', linewidth=3, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yticks(np.arange(0, 20, 2))
        ax1.set_ylim(0, 20)
        ax1.set_xticks(np.arange(10, 90, 10))
        ax1.set_xlim(10, 90)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('First Spike Latency [ms]', color=color)  # we already handled the x-label with ax1
        ax2.errorbar(fsl[ff][:, 0], fsl[ff][:, 1] * 1000, yerr=fsl[ff][:, 2] * 1000, marker='o',
                     color='r', linewidth=3, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks(np.arange(0, 20, 2))
        ax2.set_ylim(0, 20)
        plt.title(str(ff) + ' kHz')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_plot:
            # Save Plot to HDD
            figname = p + 'fi_curve_' + str(ff) + 'kHz.png'
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print('Plot saved for ' + data)
        else:
            plt.show()
        exit()

    # Plot FI-Curves
    for f in range(len(freqs)):
        plt.figure(1)
        plt.subplot(np.ceil(len(spike_count)/5), 5, f+1)
        plt.errorbar(fsl[freqs[f]][:, 0], fsl[freqs[f]][:, 1]*1000, yerr=fsl[freqs[f]][:, 2]*1000, marker='o',
                     color='r')
        plt.errorbar(spike_count[freqs[f]][:, 0], spike_count[freqs[f]][:, 1], yerr=spike_count[freqs[f]][:, 2],
                     marker='o', color='k')
        plt.ylim(0, 20)
        # plt.xlim(20, 90)
        plt.xticks(np.arange(20, 100, 10))
        plt.title(str(freqs[f]) + ' kHz')
        plt.tight_layout()
    # plt.show()

    if save_plot:
        # Save Plot to HDD
        figname = p + 'fi_curves.png'
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Plot saved for ' + data)
    else:
        plt.show()

    # Save FI to HDD
    np.save(p + 'fi_spike_count.npy', spike_count)
    np.save(p + 'fi_field.npy', fi_field)
    np.save(p + 'fi_firstspikelatency.npy', fsl)
    np.save(p + 'fi_frequencies.npy', freqs)

if Bootstrapping:
    # mf.resampling(datasets)
    nresamples = 10000
    mf.bootstrapping_vs(path_names, nresamples, plot_histogram=True)

if SOUND:  # Stimuli = Calls
    spikes = mf.spike_times_calls(path_names, 'Calls', show=False, save_data=True, th_factor=4, filter_on=True,
                                  window=None)

if EPULSES:
    method = 'exp'
    r = Parallel(n_jobs=-2)(delayed(mf.trains_to_e_pulses)(path_names, taus[k] / 1000, 0,dt_factor, stim_type=stim_type
                                                           , whole_train=True, method=method) for k in range(len(taus)))
    print('Converting done')

if VANROSSUM:
    # Try to load e pulses from HDD
    p = path_names[1]

    # Compute VanRossum Distances
    correct = np.zeros((len(duration), len(taus)))
    dist_profs = {}
    for tt in tqdm(range(len(taus)), desc='taus', leave=False):
        try:
            # Load e-pulses if available:
            trains = np.load(p + 'e_trains_' + str(taus[tt]) + '_' + stim_type + '.npy').item()
            stimulus_tags = np.load(p + 'stimulus_tags_' + str(taus[tt]) + '_' + stim_type + '.npy')
            print('Loading e-pulses from HDD done')
        except FileNotFoundError:
            # Compute e pulses if not available
            print('Could not find e-pulses, will try to compute it on the fly')
            trains, stimulus_tags = mf.trains_to_e_pulses(datasets[0], taus[tt]/1000, np.max(duration)/1000, dt_factor,
                                                          stim_type=stim_type, whole_train=whole_train, method=method)
        distances = [[]] * len(duration)
        # Parallel loop through all durations for a given tau
        r = Parallel(n_jobs=-2)(delayed(mf.vanrossum_matrix)(datasets[0], trains, stimulus_tags, duration[dur]/1000, dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=True) for dur in range(len(duration)))

        # Put values from parallel loop into correct variables
        for q in range(len(duration)):
            correct[q, tt] = r[q][1]
            distances[q] = r[q][2]
        dist_profs.update({taus[tt]: distances})

    # Save to HDD
    np.save(p + 'VanRossum_' + stim_type + '.npy', dist_profs)
    np.save(p + 'VanRossum_correct_' + stim_type + '.npy', correct)

    print('VanRossum Distances done')

if PLOT_CORRECT:
    save_plot = True
    plot_vanrossum_matrix = False
    p = path_names[1]
    correct = np.load(p + 'distances_correct_' + stim_type + '.npy')
    vr = np.load(p + 'VanRossum_correct_' + stim_type + '.npy')
    # high_taus = np.load(p + 'VanRossum_correct_hightaus.npy')

    profs = profs_plot_correct

    # Add missing cols to vanrossum
    # vanrossum = np.c_[vr, high_taus]
    vanrossum = vr

    if plot_vanrossum_matrix:
        # Plot Vanrossum Matrix
        fig, ax = plt.subplots()
        matrix = ax.pcolor(vanrossum.transpose(), vmin=0, vmax=1)
        plt.xlabel('Duration [ms]')
        plt.ylabel('Tau [ms]')
        fig.colorbar(matrix, orientation='vertical', fraction=0.04, pad=0.02)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(len(duration)) + 0.5, minor=False)
        ax.set_yticks(np.arange(len(taus)) + 0.5, minor=False)
        # ax.invert_yaxis()

        # Set correct labels
        ax.set_xticklabels(duration, minor=False)
        ax.set_yticklabels(taus, minor=False)

        if save_plot:
            # Save Plot to HDD
            figname = p + 'VanRossum_TausAndDistMatrix_' + stim_type + '.png'
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print('Plot saved')
        else:
            plt.show()

    # Plot all parameter free distances
    for k in range(len(profs)):
        if profs[k] == 'VanRossum':
            plt.subplot(np.ceil(len(profs) / 3), 3, k + 1)
            for i in range(vanrossum.shape[1]):
                plt.plot(duration, vanrossum[:, i], 'o-')
            plt.xlabel('Spike Train Length [ms]')
            plt.ylabel('Correct')
            plt.ylim(0, 1)
            plt.title(profs[k])
            # plt.legend(taus)
        else:
            plt.subplot(np.ceil(len(profs)/3), 3, k+1)
            plt.plot(duration, correct[:, k], 'ko-')
            plt.xlabel('Spike Train Length [ms]')
            plt.ylabel('Correct')
            plt.ylim(0, 1)
            plt.title(profs[k])
    # plt.tight_layout()

    if save_plot:
        # Save Plot to HDD
        figname = p + 'Correct_all_distances_' + stim_type + '.png'
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Plot saved')
    else:
        plt.show()

if ISI:
    path_save = path_names[1]
    plot_correct = False
    save_fig = False
    dist_profs = {}

    correct = np.zeros((len(duration), len(profs)))
    for p in tqdm(range(len(profs)), desc='Profiles'):
        distances_all = [[]] * len(duration)
        # Parallel loop through all durations for a given tau
        r = Parallel(n_jobs=-2)(delayed(mf.isi_matrix)(datasets[0], duration[i]/1000, boot_sample=nsamples,
                                                       stim_type=stim_type, profile=profs[p], save_fig=save_fig) for i in range(len(duration)))

        # Put values from parallel loop into correct variables
        for q in range(len(duration)):
            correct[q, p] = r[q][1]
            distances_all[q] = r[q][2]
        dist_profs.update({profs[p]: distances_all})

    # Save to HDD
    np.save(path_save + 'distances_' + stim_type + '.npy', dist_profs)
    np.save(path_save + 'distances_correct_' + stim_type + '.npy', correct)

    if plot_correct:
        for k in range(len(profs)):
            plt.subplot(np.ceil(len(profs)/2), 2, k+1)
            plt.plot(duration, correct[:, k], 'ko-')
            plt.xlabel('Spike Train Length [ms]')
            plt.ylabel('Correct [' + profs[k] + ']')
        plt.show()

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

if PULSE_TRAIN_VANROSSUM:
    save_plot = True
    p = path_names[1]
    vanrossum = np.load(p + 'VanRossum.npy').item()
    # vanrossum[tau][duratiom][boot]

    tau = taus[2]

    if stim_type == 'moth_single':
        stim_type2 = 'naturalmothcalls'
    elif stim_type == 'moth_series':
        stim_type2 = 'callseries/moths'
    else:
        print('No Pulse Trains available for: ' + stim_type)
        exit()

    fs = 480 * 1000
    print('Computing Comparison between Pulse Trains and Spike Trains (VanRossum)')
    for q in tqdm(range(len(duration)), desc='Durations'):
        idx = np.where(np.array(duration) == duration[q])[0][0]
        duration[q] = duration[q] / 1000
        distances = vanrossum[tau][idx]
        e_pulses_dur = int(duration[q] / ((tau / 1000) / dt_factor))

        # Convert matlab files to pyhton
        calls, calls_names = mf.mattopy(stim_type2, fs)

        # Convert pulse times to e-pulses
        e_pulses = mf.pulse_trains_to_e_pulses(calls, tau/1000, dt_factor, method)

        # Compute VanRossum Matrix for Pulse Trains
        d_pulses_vr = np.zeros((len(e_pulses), len(e_pulses)))
        for k in range(len(e_pulses)):
            for i in range(len(e_pulses)):
                d_pulses_vr[k, i] = mf.vanrossum_distance(e_pulses[k][0:e_pulses_dur], e_pulses[i][0:e_pulses_dur], dt_factor, tau)
        # Compute Mean (over boots) of VanRossum for Spike Trains
        d_st_vr = distances[len(distances) - 1][0]
        for k in range(len(distances) - 1):
            d_st_vr = d_st_vr + distances[k][0]
        d_st_vr = d_st_vr / len(distances)

        # Normalize
        d_st_vr = d_st_vr / np.max(d_st_vr)
        d_pulses_vr = d_pulses_vr / np.max(d_pulses_vr)

        # Plot
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(d_pulses_vr)
        plt.clim(0, 1)
        plt.xticks(np.arange(0, len(d_pulses_vr), 1))
        plt.yticks(np.arange(0, len(d_pulses_vr), 1))
        plt.xlabel('Original Call')
        plt.ylabel('Matched Call')
        plt.colorbar(fraction=0.04, pad=0.02)
        plt.title('VanRossum Pulse Trains [' + str(duration[q] * 1000) + ' ms, tau=' + str(tau) + ' ms]')

        plt.subplot(1, 2, 2)
        plt.imshow(d_st_vr)
        plt.clim(0, 1)
        plt.xticks(np.arange(0, len(d_st_vr), 1))
        plt.yticks(np.arange(0, len(d_st_vr), 1))
        plt.xlabel('Original Call')
        plt.ylabel('Matched Call')
        plt.colorbar(fraction=0.04, pad=0.02)
        plt.title('VanRossum Spike Trains [' + str(duration[q] * 1000) + ' ms, tau=' + str(tau) + ' ms ] (boot = ' + str(nsamples) + ')')
        # plt.tight_layout()
        if save_plot:
            # Save Plot to HDD
            figname = p + 'pulseVSspike_train_VanRossum_' + str(duration[q] * 1000) + 'ms_' + str(tau) + 'ms.png'
            fig = plt.gcf()
            fig.set_size_inches(20, 10)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)
            # print('Plot saved')
        else:
            plt.show()

if FI_OVERANIMALS:
    # Load data
    '''
    p = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/DataFiles/"
    a = np.load(p + 'fi_spike_count.npy')
    b = np.load(p + 'fi_firstspikelatency.npy')
    c = np.load(p + 'fi_field.npy')
    d = np.load(p + 'fi_frequencies.npy')
    embed()
    '''
    save_plot = True
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    plt.rc('font', **font)

    fields = {}
    for i in range(len(datasets)-1):
        p = "/media/brehm/Data/MasterMoth/figs/" + datasets[i] + "/DataFiles/"
        fields.update({i: np.load(p + 'fi_field.npy')})
        plt.plot(fields[i][:, 0], fields[i][:, 1], 'o-', markersize=8, linewidth=3)
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('dB SPL at threshold')
    plt.ylim(0, 90)
    plt.yticks(np.arange(0, 90, 10))
    plt.xlim(0, 110)
    plt.xticks(np.arange(10, 110, 10))
    mf.adjustSpines(plt.gca())

    if save_plot:
        # Save Plot to HDD
        p = "/media/brehm/Data/MasterMoth/figs/"
        figname = p + 'fi_field_Carales_n4.png'
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Plot saved')
    else:
        plt.show()

print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
