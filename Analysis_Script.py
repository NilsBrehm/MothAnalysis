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

FIFIELD = True
INTERVAL_MAS = False
Bootstrapping = False
INTERVAL_REC = False
GAP = False
SOUND = False
POISSON = False

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
th_factor = 4
mph_percent = 2
bin_size = 0.001
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
            if FIFIELD:
                if row[4] == 'True' and row[6] == 'Estigmene':  # this is FI
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


# Rect Intervals
if INTERVAL_REC:
    vs_order = 2
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
        mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=True, save_data=False, old=False, vs_order=vs_order)
        exit()

    # mf.plot_cohen(protocol_name, datasets, save_fig=True)

# Analyse Intervals MothASongs data stored on HDD
if INTERVAL_MAS:
    old = False
    vs_order = 2
    protocol_name = 'intervals_mas'
    spike_detection = False
    show_detection = False
    data_name = datasets[18]
    # old 0:9
    print(data_name)
    if old:
        print('OLD MAS Protocol!')
    path_names = mf.get_directories(data_name=data_name)

    if spike_detection:
        mf.spike_times_gap(path_names, protocol_name, show=show_detection, save_data=True, th_factor=th_factor,
                           filter_on=True, window=None, mph_percent=mph_percent)

    mf.interval_analysis(path_names, protocol_name, bin_size, save_fig=True, show=[True, True], save_data=False, old=old, vs_order=vs_order)

    # OLD:
    # mf.moth_intervals_spike_detection(path_names, window=None, th_factor=th_factor, mph_percent=mph_percent,
    #                                   filter_on=True, save_data=False, show=show, bin_size=bin_size)
    # mf.moth_intervals_analysis(datasets[0])

if POISSON:
    vs_order = 2
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
        fig.set_size_inches(3.9, 1.9)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)

# Analyse FIField data stored on HDD
if FIFIELD:
    save_analysis = False
    species = 'Carales'

    if species is 'Estigmene':
        # Estigmene:
        datasets = ['2017-11-25-aa', '2017-11-27-aa']  # 20 ms
        # datasets = ['2017-10-26-aa', '2017-12-05-aa']  # 50 ms
        # datasets = ['2017-10-26-aa', '2017-11-25-aa', '2017-11-27-aa', '2017-12-05-aa']
    elif species is 'Carales':
        # Carales:
        datasets = ['2017-11-01-aa', '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']  # 20 ms
        # datasets = ['2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac']  # 50 ms
        # datasets = ['2017-10-23-ah']  # 5 ms
        # datasets = ['2017-10-23-ah', '2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac', '2017-11-01-aa',
        #             '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']

    fi = [[]] * len(datasets)
    duration = []
    for k in range(len(datasets)):
        data_set_number = k
        data_name = datasets[data_set_number]
        print(str(data_set_number+1) + ' of ' + str(len(datasets)))
        print(data_name)
        path_names = mf.get_directories(data_name=data_name)

        save_plot = True
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
            x_d, y_d, params_d, perr_d = mf.fit_function(d_isi[ff][:, 0], d_isi[ff][:, 2])
            x_r, y_r, params_r, perr_r = mf.fit_function(spike_count[ff][:, 0], spike_count[ff][:, 1])
            x_inst, y_inst, params_inst, perr_inst = mf.fit_function(instant_rate[ff][:, 0], instant_rate[ff][:, 1])
            x_conv, y_conv, params_conv, perr_conv = mf.fit_function(conv_rate[ff][:, 0], conv_rate[ff][:, 1])

            # Compute Fitting Error
            # print(str(ff) + ' kHz: Summed Error = ' + str(perr_conv[2]+perr_conv[1]))
            slope_conv = params_conv[-1]
            slope_d = params_d[-1]
            slope_inst = params_inst[-1]
            slope_r = params_r[-1]

            summed_error_conv = perr_conv[2]+perr_conv[1]
            summed_error_d = perr_d[2] + perr_d[1]
            summed_error_r = perr_r[2] + perr_r[1]
            summed_error_inst = perr_inst[2] + perr_inst[1]

            if ff == 60:
                embed()
                exit()
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
            th_d_fit = params_d[2]
            th_r_fit = params_r[2]
            th_inst_fit = params_inst[2]
            th_conv_fit = params_conv[2]

            if summed_error_d > 500:
                estimated_th_d[i] = np.max(d_isi[ff][:, 0])
            else:
                estimated_th_d[i] = th_d_fit

            if summed_error_r > 500 or slope_r <= 1.1:
                estimated_th_r[i] = np.max(spike_count[ff][:, 0])
            else:
                estimated_th_r[i] = th_r_fit

            if summed_error_inst > 500 or slope_inst <= 1.1:
                estimated_th_inst[i] = np.max(instant_rate[ff][:, 0])
            else:
                estimated_th_inst[i] = th_inst_fit

            if summed_error_conv > 500 or slope_conv <= 1.1:
                estimated_th_conv[i] = np.max(conv_rate[ff][:, 0])
            else:
                estimated_th_conv[i] = th_conv_fit

            # Plot FI Curves
            if plot_fi_curves:
                # plt.plot(d_isi[ff][:, 0], d_isi[ff][:, 1], label='isi')
                plt.subplot(2, 2, 1)
                plt.plot(d_isi[ff][:, 0], d_isi[ff][:, 2], 'ko', label='sync')
                plt.plot(x_d, y_d, 'k')
                # plt.plot(spike_count[ff][:, 0], [threshold_d] * len(spike_count[ff][:, 0]), 'r--')
                # plt.plot([th_d, th_d], [0, threshold_d], 'r--')
                plt.plot([th_d_fit, th_d_fit], [0, 1], 'r--')
                plt.title('Distance')
                plt.xlabel('Amplitude [dB SPL]')
                plt.ylabel('SYNC Value')
                plt.ylim(0, 1)
                # plt.plot(d_isi[ff][:, 0], d_isi[ff][:, 3], label='spike')

                plt.subplot(2, 2, 2)
                y_max = 30
                plt.plot(spike_count[ff][:, 0], spike_count[ff][:, 1], 'ko', label='spike_count')
                plt.plot(x_r, y_r, 'k')
                # plt.plot(spike_count[ff][:, 0], [threshold_r] * len(spike_count[ff][:, 0]), 'r--')
                # plt.plot([th_r, th_r], [0, threshold_r], 'r--')
                plt.plot([th_r_fit, th_r_fit], [0, y_max], 'g--')
                plt.xlabel('Amplitude [dB SPL]')
                plt.ylabel('Spike count')
                plt.title('spike count')
                plt.ylim(0, y_max)
                plt.yticks(np.arange(0, y_max, 2))

                plt.subplot(2, 2, 3)
                y_max = 20
                plt.plot(fsl[ff][:, 0], fsl[ff][:, 1], 'ro--', label='fs latency')
                plt.plot(fisi[ff][:, 0], fisi[ff][:, 1], 'bo--', label='fs ISI')
                plt.legend()
                plt.xlabel('Amplitude [dB SPL]')
                plt.ylabel('Time [ms]')
                plt.title('First Spike Latency and Interval')
                plt.ylim(0, y_max)
                plt.yticks(np.arange(0, y_max, 2))

                plt.subplot(2, 2, 4)
                y_max = 600
                plt.plot(instant_rate[ff][:, 0], instant_rate[ff][:, 1], 'ko', label='inst')
                plt.plot(conv_rate[ff][:, 0], conv_rate[ff][:, 1], 'ro', label='conv')
                plt.plot(x_inst, y_inst, 'k')
                plt.plot(x_conv, y_conv, 'r')
                plt.plot([th_inst_fit, th_inst_fit], [0, np.max(instant_rate[ff][:, 1])], 'k--')
                plt.plot([th_conv_fit, th_conv_fit], [0, np.max(conv_rate[ff][:, 1])], 'r--')
                plt.xlabel('Amplitude [dB SPL]')
                plt.ylabel('Firing rate [Hz]')
                plt.title('Rates: Error=' + str(summed_error_conv) + ', slope=' + str(slope_conv))
                plt.ylim(0, y_max)
                plt.yticks(np.arange(0, y_max, 100))
                plt.legend()

                plt.suptitle(str(ff) + ' kHz')
                fig = plt.gcf()
                fig.set_size_inches(10.9, 10.9)
                fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.5, hspace=0.5)
                # plt.show()
                figname = p + 'FICRUVE_' + str(ff) +'.png'
                fig.savefig(figname, bbox_inches='tight', dpi=150)
                plt.close(fig)

        if plot_fi_field:
            mf.plot_settings()
            fig, ax = plt.subplots()
            ax.plot(freqs, estimated_th_d, 'bo-', label='SYNC')
            ax.plot(freqs, estimated_th_r, 'ko-', label='Spike count')
            ax.plot(freqs, estimated_th_inst, 'go-', label='Inst rate')
            ax.plot(freqs, estimated_th_conv, 'ro-', label='Conv rate')
            # plt.plot(freqs, data_th_d, 'b:', alpha=0.5, label='Data SYNC')
            # plt.plot(freqs, data_th_r, 'k:', alpha=0.5, label='Data spike count')
            ax.legend()
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

    exit()
    idx = np.array(duration) == 0.02
    fi = np.array(fi)
    fi_20ms = fi[idx]

    np.save('/media/brehm/Data/MasterMoth/figs/' + 'fi_20ms_' + species + '.npy', fi_20ms)
    print('FI for ' + species + ' was saved')

    #         for th in ths:
    #             # th = 200  # in Hz (or spike count)
    #             spike_count, rate, fi_field, fsl, dur = mf.fifield_analysis2(path_names, th, plot_fi=False, method='rate')
    #             # freqs = np.zeros(len(spike_count))
    #             freqs = [[]] * len(spike_count)
    #             i = 0
    #             for key in spike_count:
    #                 freqs[i] = int(key)
    #                 i += 1
    #             freqs = sorted(freqs)
    #
    #             mf.plot_settings()
    #             # Plot FI Field
    #             if plot_fi_field:
    #                 plt.plot(fi_field[:, 0], fi_field[:, 1], 'o-', label='th=' + str(th))
    #                 plt.xlabel('Frequency [kHz]')
    #                 plt.ylabel('dB SPL at Threshold (' + str(th) + ' Hz)')
    #                 plt.ylim(0, 90)
    #                 plt.yticks(np.arange(0, 90, 20))
    #                 plt.xlim(10, 110)
    #                 plt.xticks(np.arange(20, 110, 10))
    #         plt.title('Stimulus Duration: ' + str(dur*1000) + ' ms')
    #         plt.legend()
    #         if save_plot:
    #             # Save Plot to HDD
    #             figname = p + 'fi_field.png'
    #             fig = plt.gcf()
    #             fig.set_size_inches(5, 5)
    #             fig.savefig(figname, bbox_inches='tight', dpi=150)
    #             plt.close(fig)
    #             print('FI Field Plot saved for ' + data)
    #         else:
    #             plt.show()
    #
    #         # Plot single FI Curve
    #         if single_fi:
    #             for ff in tqdm(freqs, desc='FI Curves'):
    #                 fig, ax1 = plt.subplots()
    #                 color = 'k'
    #                 ax1.set_xlabel('Sound Pressure Level [dB SPl]')
    #                 ax1.set_ylabel('Firing Rate [Hz]', color=color)
    #                 ax1.errorbar(rate[ff][:, 0], rate[ff][:, 1], yerr=rate[ff][:, 2],
    #                              marker='o', color='k', linewidth=1, markersize=3)
    #                 for i in ths:
    #                     ax1.plot(rate[ff][:, 0], [i] * len(rate[ff][:, 0]), 'g--')
    #                 ax1.tick_params(axis='y', labelcolor=color)
    #                 ax1.set_yticks(np.arange(0, 500, 100))
    #                 ax1.set_ylim(0, 500)
    #                 ax1.set_xticks(np.arange(10, 90, 10))
    #                 ax1.set_xlim(10, 90)
    #
    #                 # First Spike Latency
    #                 ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #                 color = 'tab:red'
    #                 ax2.set_ylabel('First Spike Latency [ms]', color=color)  # we already handled the x-label with ax1
    #                 ax2.errorbar(fsl[ff][:, 0], fsl[ff][:, 1] * 1000, yerr=fsl[ff][:, 2] * 1000, marker='o',
    #                              color='r', linewidth=0.5, markersize=2, alpha=0.5)
    #                 ax2.tick_params(axis='y', labelcolor=color)
    #                 ax2.set_yticks(np.arange(0, 20, 2))
    #                 ax2.set_ylim(0, 20)
    #
    #                 plt.title(str(ff) + ' kHz')
    #                 fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #                 if save_plot:
    #                     # Save Plot to HDD
    #                     figname = p + 'fi_curve_' + str(ff) + 'kHz.png'
    #                     fig = plt.gcf()
    #                     fig.set_size_inches(5, 5)
    #                     fig.savefig(figname, bbox_inches='tight', dpi=80)
    #                     plt.close(fig)
    #                     # print('Plot saved for ' + str(ff) + ' kHz')
    #                 else:
    #                     plt.show()
    #     except:
    #         print(data_name + ' not found')
    # print('')
    # print('datasets:')
    # for t in datasets:
    #     print(t)
    # exit()
    #
    # # Plot FI-Curves
    # for f in range(len(freqs)):
    #     plt.figure(1)
    #     plt.subplot(np.ceil(len(spike_count)/5), 5, f+1)
    #     plt.errorbar(fsl[freqs[f]][:, 0], fsl[freqs[f]][:, 1]*1000, yerr=fsl[freqs[f]][:, 2]*1000, marker='o',
    #                  color='r')
    #     plt.errorbar(spike_count[freqs[f]][:, 0], spike_count[freqs[f]][:, 1], yerr=spike_count[freqs[f]][:, 2],
    #                  marker='o', color='k')
    #     plt.ylim(0, 20)
    #     # plt.xlim(20, 90)
    #     plt.xticks(np.arange(20, 100, 10))
    #     plt.title(str(freqs[f]) + ' kHz')
    #     plt.tight_layout()
    # # plt.show()
    #
    # if save_plot:
    #     # Save Plot to HDD
    #     figname = p + 'fi_curves.png'
    #     fig = plt.gcf()
    #     fig.set_size_inches(20, 10)
    #     fig.savefig(figname, bbox_inches='tight', dpi=300)
    #     plt.close(fig)
    #     print('Plot saved for ' + data)
    # else:
    #     plt.show()
    #
    # # Save FI to HDD
    # np.save(p + 'fi_spike_count.npy', spike_count)
    # np.save(p + 'fi_field.npy', fi_field)
    # np.save(p + 'fi_firstspikelatency.npy', fsl)
    # np.save(p + 'fi_frequencies.npy', freqs)

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
    save_fig = True
    mf.plot_settings()
    fi_carales = np.load('/media/brehm/Data/MasterMoth/figs/fi_20ms_Carales.npy')
    fi_estigmene = np.load('/media/brehm/Data/MasterMoth/figs/fi_20ms_Estigmene.npy')

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    cc = ['0', '0.25', '0.5', '0.75']
    for k in range(len(fi_carales)):
        ax1.plot(fi_carales[k][:, 0], fi_carales[k][:, 1], label=str(k), color=cc[k])
    for e in range(len(fi_estigmene)):
        ax2.plot(fi_estigmene[e][:, 0], fi_estigmene[e][:, 1], label=str(e), color=cc[e])
    # ax1.legend()
    # ax2.legend()

    ax1.set_ylabel('Sound level at threshold [dB SPL]')
    fig.text(0.5, 0.05, 'Frequency [kHz]', ha='center', fontdict=None)
    ax1.text(80, 90, 'n = ' + str(len(fi_carales)))
    ax2.text(80, 90, 'n = ' + str(len(fi_estigmene)))
    ax1.set_xlim(0, 110)
    ax1.set_xticks(np.arange(0, 120, 20))
    ax1.set_ylim(20, 100)
    ax1.set_yticks(np.arange(20, 101, 10))
    ax2.set_xlim(0, 110)
    ax2.set_xticks(np.arange(0, 120, 20))
    ax2.set_ylim(20, 100)
    ax2.set_yticks(np.arange(20, 101, 10))
    sns.despine()

    if save_fig:
        # Save Plot to HDD
        p = "/media/brehm/Data/MasterMoth/figs/"
        figname = p + 'fi_field_two_species.pdf'
        fig = plt.gcf()
        fig.set_size_inches(5.9, 1.9)
        fig.subplots_adjust(left=0.3, top=0.8, bottom=0.2, right=0.7, wspace=0.4, hspace=0.8)
        fig.savefig(figname)
        plt.close(fig)
        print('Plot saved')
    else:
        plt.show()

print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
