import pyspike as spk
import scipy.io as sio
from tqdm import tqdm
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from joblib import Parallel,delayed
import time
import cProfile


def plot_settings():
    # Font:
    # matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10

    # Ticks:
    matplotlib.rcParams['xtick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.size'] = 4
    matplotlib.rcParams['xtick.major.size'] = 4

    # Title Size:
    matplotlib.rcParams['axes.titlesize'] = 10

    # Axes Label Size:
    matplotlib.rcParams['axes.labelsize'] = 10

    # Axes Line Width:
    matplotlib.rcParams['axes.linewidth'] = 1

    # Tick Label Size:
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9

    # Line Width:
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['lines.color'] = 'k'

    # Marker Size:
    matplotlib.rcParams['lines.markersize'] = 2

    # Error Bars:
    matplotlib.rcParams['errorbar.capsize'] = 0

    # Legend Font Size:
    matplotlib.rcParams['legend.fontsize'] = 6

    return matplotlib.rcParams


def get_ratios(z, max_rand, duration, stim_types, calls):
    dur = np.array(duration) / 1000
    results = np.zeros(shape=(len(dur), 5))
    results_sync = np.zeros(shape=(len(dur), 5))
    results_dur = np.zeros(shape=(len(dur), 5))
    results_count = np.zeros(shape=(len(dur), 5))

    for j in range(len(dur)):  # Loop through all durations
        # t0 = time.time()
        d = np.zeros(len(calls))
        d_sync = np.zeros(len(calls))
        d_dur = np.zeros(len(calls))
        d_count = np.zeros(len(calls))
        sp = [[]] * len(calls)
        for k in range(len(calls)):  # Loop through all calls
            # Add jitter
            calls[k] = np.sort(calls[k])
            idx = calls[k] <= dur[j]
            dummy01 = calls[k][idx]

            # settings for random jitter: from a=-max_rand to b=rand_max seconds
            a = -max_rand
            b = max_rand

            # Now add some jitter to the spike trains
            spike_times = [dummy01] * trials
            for rr in range(len(spike_times)):
                jitter = [[]] * len(dummy01)
                for jj in range(len(dummy01)):
                    if no_jitter:
                        jitter[jj] = 0
                    else:
                        jitter[jj] = (b - a) * np.random.random_sample() + a
                dd = spike_times[rr] + jitter
                dd[dd < 0] = np.abs(dd[dd < 0])
                spike_times[rr] = spk.SpikeTrain(np.sort(dd), [0, dur[j]])

            sp[k] = spike_times
            # t1 = time.time()
            # ISI and SYNC
            # d[k] = abs(spk.isi_distance(spike_times, interval=[0, dur[j]]))
            d[k] = abs(spk.isi_distance(spike_times))
            if np.max(d[k]) > 1:
                print('Warning: ISI > 1 for: ' + str(max_rand * 1000) + ' ms and dur=' + str(dur[j]))
                print('ISI d: ' + str(np.max(d)))

            # d_sync[k] = spk.spike_sync(spike_times, interval=[0, dur[j]])
            d_sync[k] = spk.spike_sync(spike_times)

            # t2 = time.time()
            # DUR and COUNT metric
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
            # t3 = time.time()

        # Overall Distances
        sp = np.concatenate(np.array(sp))
        # over_all = abs(spk.isi_distance(sp, interval=[0, dur[j]]))
        over_all = abs(spk.isi_distance(sp))
        # over_all_sync = spk.spike_sync(sp, interval=[0, dur[j]])
        over_all_sync = spk.spike_sync(sp)

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
    p = '/media/nils/Data/Moth/CallStats/pulsetrains/'
    np.save(p + 'ISI_Ratios_PulseTrains_' + stim_types + '_' + str(z) + '.npy', results)
    np.save(p + 'SYNC_Ratios_PulseTrains_' + stim_types + '_' + str(z) + '.npy', results_sync)
    np.save(p + 'DUR_Ratios_PulseTrains_' + stim_types + '_' + str(z) + '.npy', results_dur)
    np.save(p + 'COUNT_Ratios_PulseTrains_' + stim_types + '_' + str(z) + '.npy',
            results_count)


# ======================================================================================================================
# SCRIPT SECTION =======================================================================================================
# ======================================================================================================================
RATIOS = False
compute_ratios = False
plot_ratios = False

compare_ratios = True

# max_rand = 0.01
no_jitter = False
trials = 20
if no_jitter:
    rands = [0]
else:
    rands = np.round(np.geomspace(0.001, 0.1, num=100), 5)

stims = ['series', 'single']

if compute_ratios:
    # Series
    stim_type = 'series'
    durations = list(np.arange(0, 2550, 50))
    durations[0] = 10
    calls01 = sio.loadmat('/media/nils/Data/Moth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
    # cProfile.run('get_ratios(0, rands[0], durations, stim_type, calls01)')
    # exit()
    r1 = Parallel(n_jobs=3)(delayed(get_ratios)(z, rands[z], durations, stim_type, calls01) for z in tqdm(range(len(rands)), desc='Jitter'))
    print('Call Series Done')

    # Single
    stim_type = 'single'
    durations = list(np.arange(0, 255, 5))
    durations[0] = 2
    calls02 = sio.loadmat('/media/nils/Data/Moth/CallStats/samples.mat')['samples'][0]
    r2 = Parallel(n_jobs=3)(delayed(get_ratios)(z, rands[z], durations, stim_type, calls02) for z in tqdm(range(len(rands)), desc='Jitter'))

    print('Single Calls Done')


if RATIOS:
    for st in range(len(stims)):
        stim_type = stims[st]
        if stim_type is 'series':
            durations = list(np.arange(0, 2550, 50))
            durations[0] = 10
            calls01 = sio.loadmat('/media/nils/Data/Moth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
        if stim_type is 'single':
            durations = list(np.arange(0, 255, 5))
            durations[0] = 2
            calls01 = sio.loadmat('/media/nils/Data/Moth/CallStats/samples.mat')['samples'][0]

        # if compute_ratios:
        #     # get_ratios(rands[98], durations, stim_type, calls01)
        #     # exit()
        #     Parallel(n_jobs=3)(delayed(get_ratios)(z, rands[z], durations, stim_type, calls01) for z in tqdm(range(len(rands)), desc='Jitter'))

        if plot_ratios:
            for z in tqdm(range(len(rands)), desc='Jitters'):
                max_rands = rands[z]
                # ======================================================================================================
                # ISI and SYNC =========================================================================================
                p = '/media/nils/Data/Moth/CallStats/pulsetrains/'

                ratios_isi = np.load(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy')
                ratios_sync = np.load(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy')

                # Plot
                plot_settings()
                if stim_type == 'series':
                    x_end = 2500 + 100
                    x_step = 500
                if stim_type == 'single':
                    x_end = 250 + 10
                    x_step = 50

                # Create Grid
                grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
                fig = plt.figure(figsize=(5.9, 2.9))
                ax1 = plt.subplot(grid[0])
                ax2 = plt.subplot(grid[1])
                ax3 = plt.subplot(grid[2])
                ax4 = plt.subplot(grid[3])

                # ax1.errorbar(durations, ratios_isi[:, 0], yerr=ratios_isi[:, 1], color='k', marker='o', label='within')
                ax1.plot(durations, ratios_isi[:, 0], 'k-')
                ax1.fill_between(durations, ratios_isi[:, 0] - ratios_isi[:, 1],
                                 ratios_isi[:, 0] + ratios_isi[:, 1], facecolors='k', alpha=0.25)

                ax1.plot(durations, ratios_isi[:, 2], '-', label='between', color='blue')
                ax1.set_ylim(0, 1)
                ax1.set_yticks(np.arange(0, 1.1, 0.2))
                ax1.set_xticklabels([])
                ax1.set_ylabel('ISI')
                ax1.set_xlim(0, x_end)
                ax1.set_xticks(np.arange(0, x_end, x_step))

                ax3.plot(durations, ratios_isi[:, 3], 'r-', label='ratio')
                ax3.set_ylim(0, 3)
                ax3.set_yticks(np.arange(0, 3.1, 0.5))
                ax3.set_ylabel('Ratio')
                ax3.set_xlim(0, x_end)
                ax3.set_xticks(np.arange(0, x_end, x_step))

                ax2.plot(durations, ratios_sync[:, 0], 'k-')
                ax2.fill_between(durations, ratios_sync[:, 0] - ratios_sync[:, 1],
                                 ratios_sync[:, 0] + ratios_sync[:, 1], facecolors='k', alpha=0.25)
                # ax2.errorbar(durations, ratios_sync[:, 0], yerr=ratios_sync[:, 1], color='k', marker='o', label='within')
                ax2.plot(durations, ratios_sync[:, 2], '-', label='between', color='blue')
                ax2.set_ylim(0, 1)
                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.set_ylabel('SYNC')
                ax2.set_xlim(0, x_end)
                ax2.set_xticks(np.arange(0, x_end, x_step))

                ax4.plot(durations, 1/ratios_sync[:, 3], 'r-', label='ratio')
                ax4.set_ylim(0, 3)
                ax4.set_yticks(np.arange(0, 3.1, 0.5))
                ax4.set_yticklabels([])
                ax4.set_xlim(0, x_end)
                ax4.set_xticks(np.arange(0, x_end, x_step))

                # Axes Labels
                fig.text(0.5, 0.055, 'Spike train durations [ms]', ha='center', fontdict=None)

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
                figname = '/media/nils/Data/Moth/CallStats/pulsetrains/figs/Distance_Ratios_ISI_SYNC_' +\
                          stim_type + '_' + str(z) + '_' + str(int(max_rands*1000)) + '.pdf'
                fig.savefig(figname)
                plt.close(fig)
                # print('ISI and SYNC plots saved for: ' + str(max_rands*1000) + ' ms')

                # ======================================================================================================
                # DUR and COUNT ========================================================================================
                p = '/media/nils/Data/Moth/CallStats/pulsetrains/'
                ratios_isi = np.load(p + 'DUR_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy')
                ratios_sync = np.load(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy')

                # Normalize
                max_norm = np.max(ratios_isi[:, 2] * 1000)
                ratios_isi[:, 0] = (ratios_isi[:, 0] * 1000) / max_norm
                ratios_isi[:, 1] = (ratios_isi[:, 1] * 1000) / max_norm
                ratios_isi[:, 2] = (ratios_isi[:, 2] * 1000) / max_norm

                max_norm = np.max(ratios_sync[:, 2] * 1000)
                ratios_sync[:, 0] = (ratios_sync[:, 0] * 1000) / max_norm
                ratios_sync[:, 1] = (ratios_sync[:, 1] * 1000) / max_norm
                ratios_sync[:, 2] = (ratios_sync[:, 2] * 1000) / max_norm

                # Plot
                plot_settings()
                if stim_type == 'series':
                    x_end = 2500 + 100
                    x_step = 500
                if stim_type == 'single':
                    x_end = 250 + 10
                    x_step = 50

                # Create Grid
                grid = matplotlib.gridspec.GridSpec(nrows=2, ncols=2)
                fig = plt.figure(figsize=(5.9, 2.9))
                ax1 = plt.subplot(grid[0])
                ax2 = plt.subplot(grid[1])
                ax3 = plt.subplot(grid[2])
                ax4 = plt.subplot(grid[3])

                ax1.plot(durations, ratios_isi[:, 0], 'k-')
                ax1.fill_between(durations, ratios_isi[:, 0] - ratios_isi[:, 1],
                                 ratios_isi[:, 0] + ratios_isi[:, 1], facecolors='k', alpha=0.25)
                # ax1.errorbar(durations, ratios_isi[:, 0], yerr=ratios_isi[:, 1], color='k', marker='o', label='within')
                ax1.plot(durations, ratios_isi[:, 2], '-', label='between', color='blue')
                ax1.set_ylim(0, 1)
                ax1.set_yticks(np.arange(0, 1.1, 0.2))
                ax1.set_xticklabels([])
                ax1.set_ylabel('Norm. DUR')
                ax1.set_xlim(0, x_end)
                ax1.set_xticks(np.arange(0, x_end, x_step))

                ax3.plot(durations, ratios_isi[:, 3], 'r-', label='ratio')
                ax3.set_ylim(0, 3)
                ax3.set_yticks(np.arange(0, 3.1, 0.5))
                ax3.set_ylabel('Ratio')
                ax3.set_xlim(0, x_end)
                ax3.set_xticks(np.arange(0, x_end, x_step))

                ax2.plot(durations, ratios_sync[:, 0], 'k-')
                ax2.fill_between(durations, ratios_sync[:, 0] - ratios_sync[:, 1],
                                 ratios_sync[:, 0] + ratios_sync[:, 1], facecolors='k', alpha=0.25)
                # ax2.errorbar(durations, ratios_sync[:, 0], yerr=ratios_sync[:, 1], color='k', marker='o', label='within')
                ax2.plot(durations, ratios_sync[:, 2], '-', label='between', color='blue')
                ax2.set_ylim(0, 1)
                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.set_ylabel('Norm. COUNT')
                ax2.set_xlim(0, x_end)
                ax2.set_xticks(np.arange(0, x_end, x_step))

                ax4.plot(durations, ratios_sync[:, 3], 'r-', label='ratio')
                ax4.set_ylim(0, 3)
                ax4.set_yticks(np.arange(0, 3.1, 0.5))
                ax4.set_yticklabels([])
                ax4.set_xlim(0, x_end)
                ax4.set_xticks(np.arange(0, x_end, x_step))

                # Axes Labels
                fig.text(0.5, 0.055, 'Spike train durations [ms]', ha='center', fontdict=None)

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
                figname = '/media/nils/Data/Moth/CallStats/pulsetrains/figs/Distance_Ratios_DUR_COUNT_' + stim_type + \
                          '_' + str(z) + '_' + str(int(max_rands*1000)) + '.pdf'
                fig.savefig(figname)
                plt.close(fig)
                # print('DUR and COUNT plots saved for: ' + str(z) + ' ms')
        print('Plots saved')

# Compare rands
# rands = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
if compare_ratios:
        # rands = np.linspace(0.001, 0.02, 51)
        duration_single = list(np.arange(0, 255, 5))
        duration_single[0] = 2
        duration_single = np.array(duration_single) / 1000

        duration_series = list(np.arange(0, 2550, 50))
        duration_series[0] = 10
        duration_series = np.array(duration_series) / 1000

        p = '/media/nils/Data/Moth/CallStats/pulsetrains/'
        r_isi_series = [[]] * len(rands)
        r_sync_series = [[]] * len(rands)
        r_dur_series = [[]] * len(rands)
        r_count_series = [[]] * len(rands)
        r_isi_single = [[]] * len(rands)
        r_sync_single = [[]] * len(rands)
        r_dur_single = [[]] * len(rands)
        r_count_single = [[]] * len(rands)

        # Get all ratios
        for i in range(len(rands)):
            mode = 3
            stim_type = 'series'
            r_isi_series[i] = np.load(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_sync_series[i] = 1/np.load(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_dur_series[i] = np.load(p + 'DUR_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_count_series[i] = np.load(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]

            idx_inf = r_count_series[i] == np.inf
            r_count_series[i][idx_inf] = 500
            idx_inf = r_dur_series[i] == np.inf
            r_dur_series[i][idx_inf] = 500

            stim_type = 'single'
            r_isi_single[i] = np.load(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_sync_single[i] = 1/np.load(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_dur_single[i] = np.load(p + 'DUR_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            r_count_single[i] = np.load(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')[:, mode]
            idx_inf = r_count_single[i] == np.inf
            r_count_single[i][idx_inf] = 500
            idx_inf = r_dur_single[i] == np.inf
            r_dur_single[i][idx_inf] = 500

        # Plot
        # Create Grid: Duration x Jitter
        plot_settings()
        x_series = duration_series
        x_single = duration_single
        XX_series, YY_series = np.meshgrid(x_series, rands*1000)
        XX_single, YY_single = np.meshgrid(x_single, rands*1000)

        fig = plt.figure(figsize=(5.9, 3.5))
        grid = matplotlib.gridspec.GridSpec(nrows=24, ncols=52)

        ax1 = plt.subplot(grid[0:10, 0:10])
        ax2 = plt.subplot(grid[0:10, 12:22])
        ax3 = plt.subplot(grid[0:10, 24:34])
        ax4 = plt.subplot(grid[0:10, 36:46])
        ax5 = plt.subplot(grid[13:23, 0:10])
        ax6 = plt.subplot(grid[13:23, 12:22])
        ax7 = plt.subplot(grid[13:23, 24:34])
        ax8 = plt.subplot(grid[13:23, 36:46])

        cb1 = plt.subplot(grid[0:23, 50:51])
        # Ratio limits
        ratio_lower = 1
        ratio_upper = 3
        im1 = ax1.pcolormesh(XX_series, YY_series, r_isi_series, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im2 = ax2.pcolormesh(XX_series, YY_series, r_sync_series, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im3 = ax3.pcolormesh(XX_series, YY_series, r_dur_series, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im4 = ax4.pcolormesh(XX_series, YY_series, r_count_series, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)

        im5 = ax5.pcolormesh(XX_single, YY_single, r_isi_single, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im6 = ax6.pcolormesh(XX_single, YY_single, r_sync_single, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im7 = ax7.pcolormesh(XX_single, YY_single, r_dur_single, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)
        im8 = ax8.pcolormesh(XX_single, YY_single, r_count_single, cmap='jet', vmin=ratio_lower, vmax=ratio_upper, shading='flat', rasterized=True)

        # Colorbar
        c1 = matplotlib.colorbar.ColorbarBase(cb1, cmap='jet', norm=matplotlib.colors.Normalize(vmin=ratio_lower, vmax=ratio_upper),
                                              orientation='vertical', ticklocation='right')
        # c1.set_label('Ratio', rotation=-90)
        c1.set_ticks([1, 2, 3, 4])
        c1.set_ticklabels([1, 2, 3, 4])
        c1.solids.set_rasterized(True)  # Removes white lines

        # Subfig Caps
        subfig_color = 'white'
        subfig_caps = 12
        label_x_pos = 0.05
        label_y_pos = 0.83
        subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax5.text(label_x_pos, label_y_pos, subfig_caps_labels[4], transform=ax5.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax6.text(label_x_pos, label_y_pos, subfig_caps_labels[5], transform=ax6.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax7.text(label_x_pos, label_y_pos, subfig_caps_labels[6], transform=ax7.transAxes, size=subfig_caps,
                     color=subfig_color)
        ax8.text(label_x_pos, label_y_pos, subfig_caps_labels[7], transform=ax8.transAxes, size=subfig_caps,
                     color=subfig_color)
        # Axis label
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax6.set_yticklabels([])
        ax7.set_yticklabels([])
        ax8.set_yticklabels([])

        ax1.set_xticks(np.arange(0, 3, 1))
        ax1.set_xticks(np.arange(0, 3, 0.5), minor=True)
        ax2.set_xticks(np.arange(0, 3, 1))
        ax2.set_xticks(np.arange(0, 3, 0.5), minor=True)
        ax3.set_xticks(np.arange(0, 3, 1))
        ax3.set_xticks(np.arange(0, 3, 0.5), minor=True)
        ax4.set_xticks(np.arange(0, 3, 1))
        ax4.set_xticks(np.arange(0, 3, 0.5), minor=True)

        ax5.set_xticks(np.arange(0, 0.3, 0.1))
        ax5.set_xticks(np.arange(0, 0.3, 0.05), minor=True)
        ax6.set_xticks(np.arange(0, 0.3, 0.1))
        ax6.set_xticks(np.arange(0, 0.3, 0.05), minor=True)
        ax7.set_xticks(np.arange(0, 0.3, 0.1))
        ax7.set_xticks(np.arange(0, 0.3, 0.05), minor=True)
        ax8.set_xticks(np.arange(0, 0.3, 0.1))
        ax8.set_xticks(np.arange(0, 0.3, 0.05), minor=True)

        # ax1.set_yticks([1, 5, 10, 15, 20])
        y_upper_limit = 81
        ax1.set_ylim(0, y_upper_limit-1)
        ax2.set_ylim(0, y_upper_limit - 1)
        ax3.set_ylim(0, y_upper_limit - 1)
        ax4.set_ylim(0, y_upper_limit - 1)
        ax5.set_ylim(0, y_upper_limit - 1)
        ax6.set_ylim(0, y_upper_limit - 1)
        ax7.set_ylim(0, y_upper_limit - 1)
        ax8.set_ylim(0, y_upper_limit - 1)

        ax1.set_yticks(np.arange(0, y_upper_limit, 20))
        ax2.set_yticks(np.arange(0, y_upper_limit, 20))
        ax3.set_yticks(np.arange(0, y_upper_limit, 20))
        ax5.set_yticks(np.arange(0, y_upper_limit, 20))
        ax6.set_yticks(np.arange(0, y_upper_limit, 20))
        ax7.set_yticks(np.arange(0, y_upper_limit, 20))
        ax8.set_yticks(np.arange(0, y_upper_limit, 20))

        ax1.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax2.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax3.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax4.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax5.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax6.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax7.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)
        ax8.set_yticks(np.arange(0, y_upper_limit, 10), minor=True)


        grid_color = 'k'
        grid_linewidth = 0.5
        ax1.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax2.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax3.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax4.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax5.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax6.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax7.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)
        ax8.grid(which='both', color=grid_color, linestyle=':', linewidth=grid_linewidth)

        ax1.set_title('ISI', size=10)
        ax2.set_title('SYNC', size=10)
        ax3.set_title('Norm. DUR', size=10)
        ax4.set_title('Norm. COUNT', size=10)

        fig.text(0.93, 0.6, 'Ratio', ha='center', fontdict=None, rotation=-90)
        fig.text(0.83, 0.83, 'Call series', ha='center', fontdict=None, rotation=-90)
        fig.text(0.83, 0.45, 'Single calls', ha='center', fontdict=None, rotation=-90)
        fig.text(0.5, 0.075, 'Pulse train duration [s]', ha='center', fontdict=None)
        fig.text(0.02, 0.75, 'Amount of jitter [ms]', ha='center', fontdict=None, rotation=90)
        fig.subplots_adjust(left=0.1, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.1)

        figname = '/media/nils/Data/Moth/CallStats/pulsetrains/figs/pulse_train_distance_ratios2.pdf'
        fig.savefig(figname)
        plt.close(fig)
print('All done!')
