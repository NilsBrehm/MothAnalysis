import pyspike as spk
import scipy.io as sio
from tqdm import tqdm
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


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


RATIOS = False
compute_ratios = True
plot_ratios = False

compare_ratios = True

# max_rand = 0.01
no_jitter = False
trials = 20

# stim_type = 'series'
# duration = list(np.arange(0, 2550, 50))
# duration[0] = 10

stim_type = 'single'
duration = list(np.arange(0, 255, 5))
duration[0] = 2

# rands = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
# rands = [0.003, 0.004, 0.006, 0.007, 0.008, 0.009, 0.011, 0.012, 0.013, 0.014, 0.15, 0.016, 0.017, 0.018, 0.019]
rands = np.linspace(0.001, 0.02, 51)

if RATIOS:
    for z in tqdm(range(len(rands)), desc='Jitters'):
        max_rand = rands[z]
        if compute_ratios:
            # Load call series
            # calls2 = sio.loadmat('/media/nils/Data/Moth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
            calls = sio.loadmat('/media/nils/Data/Moth/CallStats/samples.mat')['samples'][0]

            dur = np.array(duration) / 1000
            results = np.zeros(shape=(len(dur), 5))
            results_sync = np.zeros(shape=(len(dur), 5))
            results_dur = np.zeros(shape=(len(dur), 5))
            results_count = np.zeros(shape=(len(dur), 5))

            for j in tqdm(range(len(dur)), desc='Distances'):  # Loop through all durations
                edges = [0, dur[j]]
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
                    # jitter_calls = [[]] * trials
                    # for rr in range(trials):
                    #     jitter = (b - a) * np.random.random_sample() + a
                    #     dummy02 = dummy01 + jitter
                    #     # If time point is negative reset it to original
                    #     dummy02[dummy02 < 0] = dummy01[dummy02 < 0]
                    #     # jitter_calls[rr] = dummy02[dummy02 <= dur[j]]

                    spike_times = [dummy01] * trials
                    jitter_calls = [dummy01] * trials
                    for rr in range(len(spike_times)):
                        jitter = [[]] * len(dummy01)
                        for jj in range(len(dummy01)):
                            if no_jitter:
                                jitter[jj] = 0
                            else:
                                jitter[jj] = (b - a) * np.random.random_sample() + a
                        dd = spike_times[rr] + jitter
                        dd[dd < 0] = dummy01[dd < 0]
                        jitter_calls[rr] = dd
                        spike_times[rr] = spk.SpikeTrain(dd,  [0, 10])

                    sp[k] = spike_times
                    d[k] = abs(spk.isi_distance(spike_times))
                    if np.max(d[k]) > 1:
                        print('Warning: ISI > 1 for: ' + str(max_rand*1000) + ' ms and dur=' + str(dur[j]))
                        print('ISI d: ' + str(np.max(d)))
                    d_sync[k] = spk.spike_sync(spike_times)
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

                sp = np.concatenate(np.array(sp))
                over_all = abs(spk.isi_distance(sp))
                over_all_sync = spk.spike_sync(sp)
                # over_all = abs(spk.spike_distance(sp, interval=[0, dur[j]]))
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
            np.save(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy', results)
            np.save(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy', results_sync)
            np.save(p + 'DUR_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy', results_dur)
            np.save(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(z) + '.npy',
                    results_count)
            # np.save(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy', results)
            # np.save(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy', results_sync)
            # np.save(p + 'DUR_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy', results_dur)
            # np.save(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy', results_count)

            print('Pulse Train Ratios saved for:' + str(max_rand*1000) + ' ms')

        if plot_ratios:
            p = '/media/nils/Data/Moth/CallStats/pulsetrains/'

            # ratios_isi = np.load(p + 'DUR_Ratios_PulseTrains_' + stim_type + '.npy')
            # ratios_sync = np.load(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '.npy')

            ratios_isi = np.load(p + 'COUNT_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy')
            ratios_sync = np.load(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(int(max_rand*1000)) + '.npy')

            # Normalize
            max_norm = np.max(ratios_isi[:, 2] * 1000)
            ratios_isi[:, 0] = (ratios_isi[:, 0] * 1000) / max_norm
            ratios_isi[:, 1] = (ratios_isi[:, 1] * 1000) / max_norm
            ratios_isi[:, 2] = (ratios_isi[:, 2] * 1000) / max_norm

            # max_norm = np.max(ratios_sync[:, 2] * 1000)
            # ratios_sync[:, 0] = (ratios_sync[:, 0] * 1000) / max_norm
            # ratios_sync[:, 1] = (ratios_sync[:, 1] * 1000) / max_norm
            # ratios_sync[:, 2] = (ratios_sync[:, 2] * 1000) / max_norm

            # Plot
            plot_settings()
            stim_length = 'series'
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
            ax1.set_ylim(0, 1)
            ax1.set_yticks(np.arange(0, 1.1, 0.2))
            ax1.set_xticklabels([])
            ax1.set_ylabel('ISI Distance')
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
            ax2.set_ylim(0, 1)
            ax2.set_yticks(np.arange(0, 1.1, 0.2))
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_ylabel('SYNC Value')
            ax2.set_xlim(0, x_end)
            ax2.set_xticks(np.arange(0, x_end, x_step))

            ax4.plot(duration, 1 / ratios_sync[:, 3], 'r-', label='ratio')
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
            figname = '/media/nils/Data/Moth/CallStats/pulsetrains/Distance_Ratios_ISI_SYNC_' + stim_type + '_' + str(int(max_rand*1000)) + '.pdf'
            fig.savefig(figname)
            plt.close(fig)
            print('Plot saved for: ' + str(max_rand*1000) + ' ms')

# Compare rands
# rands = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

if compare_ratios:
    # rands = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.15, 0.016, 0.017, 0.018, 0.019, 0.02]
    rands = np.linspace(0.001, 0.02, 51)
    p = '/media/nils/Data/Moth/CallStats/pulsetrains/'
    ratios_isi = [[]] * len(rands)
    ratios_sync = [[]] * len(rands)
    rat = [[]] * len(rands)
    for i in range(len(rands)):
        ratios_isi[i] = np.load(p + 'ISI_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')
        ratios_sync[i] = np.load(p + 'SYNC_Ratios_PulseTrains_' + stim_type + '_' + str(i) + '.npy')
        # idx = ratios_isi[i] < 1
        # ratios_isi[i][idx] = 0
        rat[i] = ratios_isi[i][:, 3]

    # rat = np.array(rat).T
    # Plot
    x = duration
    XX, YY = np.meshgrid(x, rands*1000)

    fig = plt.figure(figsize=(5.9, 2.9))
    grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=46)

    ax1 = plt.subplot(grid[0, 0:40])
    cb1 = plt.subplot(grid[0, 42:45])
    import matplotlib.colors as colors
    # im1 = ax1.imshow(rat, cmap='jet', vmin=0, vmax=3, origin='lower', interpolation='gaussian')
    # im1 = ax1.pcolormesh(XX, YY, rat, cmap='seismic', vmin=0, vmax=2, shading='gouraud')
    im1 = ax1.pcolormesh(XX, YY, rat, cmap='seismic', vmin=0, vmax=2, shading='flat')
    # im1 = ax1.pcolormesh(XX, YY, rat, cmap='jet', shading='flat', norm=colors.LogNorm(vmin=0.1, vmax=3))


    # im1 = ax1.imshow(rat, vmin=0, vmax=3, cmap='jet', origin='lower')
    # c1 = matplotlib.colorbar.ColorbarBase(cb1, cmap='jet', norm=colors.LogNorm(vmin=0.1, vmax=3),
    #                                       orientation='vertical', ticklocation='right')
    c1 = matplotlib.colorbar.ColorbarBase(cb1, cmap='seismic', norm=matplotlib.colors.Normalize(vmin=0, vmax=2),
                                          orientation='vertical', ticklocation='right')
    c1.set_label('Ratio')
    c1.set_ticks([0, 1, 2, 3])
    c1.set_ticklabels([0, 1, 2, 3])

    # ax1.set_xscale('log')
    # ax1.set_yscale('log')

    # ax1.set_ylim(rands[0], 0.02)

    # ax1.plot(x, np.log10(x)/np.max(np.log10(x))*0.02-0.01, 'r--')
    # ax1.grid(color='0.3', linestyle='--', linewidth=.5)
    #
    # th = np.copy(rat)
    # th[th <= 1] = 0
    # th[th > 1] = 1
    # im2 = ax1.pcolormesh(XX, YY, th, cmap='jet', vmin=0, vmax=1, shading='gouraud', alpha=0.5)

    # ax1.set_xticklabels(duration)
    # ax1.set_xticks(np.arange(0, len(duration), 10))
    plt.show()
