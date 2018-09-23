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
import matplotlib.patheffects as pe
from tqdm import tqdm

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


def fit_function(x_fit, data, x_plot, method):
    if method is 'limited':
        def func(xx, S, a, k):
            return S - a*np.exp(-k*xx)
    elif method is 'logistic':
        def func(xx, S, a, k):
            return (a*S) / (a + (S-a)*np.exp(-S*k*xx))
    elif method is 'power':
        def func(xx, a, b, c):
            return a*xx**b + c
    elif method is 'linear':
        def func(xx, a, c):
            return a*xx + c
    elif method is 'exp':
        def func(xx, a, c, k):
            return a*np.exp(k*xx) + c
    elif method is 'exp_decay':
        def func(xx, a, c, k):
            return a * np.exp(-xx/k) + c
    elif method is 'log':
        def func(xx, a, c, k):
            return a * np.log10(xx) + c
    else:
        print('Method not available!')
        return 0, 0, 0, 0, 0

    # p0 = [100, 1000]
    # bounds = (0, [np.max(data)/2+10, np.max(data)*2+10, np.max(x_fit), np.inf])
    popt, pcov = curve_fit(func, x_fit, data, maxfev=2000)
    x = np.linspace(x_plot[0], x_plot[1], 1000)
    y = func(x, *popt)
    y0 = func(popt[-2], *popt)
    perr = np.sqrt(np.diag(pcov))
    return x, y, popt, perr, y0

# ======================================================================================================================
# SCRIPT STARTS HERE ===================================================================================================
rec = '2018-02-16-aa'
path_names = mf.get_directories(rec)
compute_poisson = False
plot_poisson = True
TAU = False
if TAU:
    trials = 10
    rate = 100
    durations = np.arange(0, 1000, 10)
    durations[0] = 1
    durations = durations / 1000
    # base_tmax = [0.01, 0.1, 0.5, 1, 2]
    # base_tmax = [0.1, 0.5, 1, 2, 4, 8]
    base_tmax = [2, 2, 2, 2, 2]

    taus = np.arange(1, 1000, 10) / 1000
    dt = 0.001
    vr = np.ndarray((len(base_tmax), len(taus)))
    vr_expected = np.ndarray((len(base_tmax), len(taus)))

    p01 = [[]] * len(base_tmax)
    p02 = [[]] * len(base_tmax)
    for k in range(len(base_tmax)):
        p01[k], _ = poission_spikes(1, rate, tmax=base_tmax[k])
        p02[k], _ = poission_spikes(1, rate, tmax=base_tmax[k])
        for j in tqdm(range(len(taus)), desc='Taus'):
            pulse01 = mf.fast_e_pulses(np.array(p01[k][0]), tau=taus[j], dt=dt)
            pulse02 = mf.fast_e_pulses(np.array(p02[k][0]), tau=taus[j], dt=dt)
            vr_expected[k][j] = rate * base_tmax[k]
            vr[k][j] = mf.vanrossum_distance(pulse01, pulse02, dt=dt, tau=taus[j])
    embed()
    exit()

if compute_poisson:
    boots = 10
    rate = 100
    # base_tmax = [0.01, 0.1, 0.5, 1, 2]
    base_tmax = [0.05, 0.1, 0.5, 1, 2]
    t_diffs = np.arange(0.001, 2, 0.001)
    t_diffs = np.insert(t_diffs, 0, 0)
    taus = [0.001, 0.01, 0.1]
    dt = 0.001
    tau = 0.1
    data = []

    vr = np.ndarray((len(base_tmax), len(t_diffs), len(taus)))
    isi = np.ndarray((len(base_tmax), len(t_diffs)))
    sync = np.ndarray((len(base_tmax), len(t_diffs)))
    count = np.ndarray((len(base_tmax), len(t_diffs)))
    dur = np.ndarray((len(base_tmax), len(t_diffs)))
    spike = np.ndarray((len(base_tmax), len(t_diffs)))

    # pp01, _ = poission_spikes(1, rate, tmax=base_tmax[-1])
    # pp02, _ = poission_spikes(1, rate, tmax=base_tmax[-1] + t_diffs[-1])
    # pp01 = np.array(pp01[0])
    # pp02 = np.array(pp02[0])

    for k in tqdm(range(len(base_tmax)), desc='Poisson Trains: '):
        for i in range(len(t_diffs)):
            p01, _ = poission_spikes(1, rate, tmax=base_tmax[k])
            p02, _ = poission_spikes(1, rate, tmax=base_tmax[k]+t_diffs[i])
            p01 = p01[0]
            p02 = p02[0]
            # p01 = pp01[pp01 <= base_tmax[k]]
            # p02 = pp02[pp02 <= base_tmax[k] + t_diffs[i]]
            spike_train01 = spk.SpikeTrain(p01, [0, base_tmax[k]+t_diffs[i]])
            spike_train02 = spk.SpikeTrain(p02, [0, base_tmax[k]+t_diffs[i]])
            for j in range(len(taus)):
                pulse01 = mf.fast_e_pulses(np.array(p01), tau=taus[j], dt=dt)
                pulse02 = mf.fast_e_pulses(np.array(p02), tau=taus[j], dt=dt)
                vr[k][i][j] = mf.vanrossum_distance(pulse01, pulse02, dt=dt, tau=taus[j])
            isi[k][i] = spk.isi_distance(spike_train01, spike_train02)
            sync[k][i] = spk.spike_sync(spike_train01, spike_train02)
            count[k][i] = abs(len(spike_train01.spikes) - len(spike_train02.spikes))
            dur[k][i] = abs(spike_train01.spikes[-1] - spike_train02[-1])
            spike[k][i] = spk.spike_distance(spike_train01, spike_train02)
    # FIT
    fit_vr_x = np.ndarray((len(base_tmax), len(taus), 1000))
    fit_vr_y = np.ndarray((len(base_tmax), len(taus), 1000))
    fit_isi_x = np.ndarray((len(base_tmax), 1000))
    fit_isi_y = np.ndarray((len(base_tmax), 1000))
    fit_sync_x = np.ndarray((len(base_tmax), 1000))
    fit_sync_y = np.ndarray((len(base_tmax), 1000))
    fit_count_x = np.ndarray((len(base_tmax), 1000))
    fit_count_y = np.ndarray((len(base_tmax), 1000))
    fit_dur_x = np.ndarray((len(base_tmax), 1000))
    fit_dur_y = np.ndarray((len(base_tmax), 1000))
    fit_spike_x = np.ndarray((len(base_tmax), 1000))
    fit_spike_y = np.ndarray((len(base_tmax), 1000))

    for j in tqdm(range(len(base_tmax)), desc='Fit: '):
        for t in range(len(taus)):
            fit_vr_x[j][t], fit_vr_y[j][t], _, _, _ = fit_function(t_diffs, vr[j][:, t], x_plot=[t_diffs[0], t_diffs[-1]],
                                                                   method='power')

        fit_isi_x[j], fit_isi_y[j], _, _, _ = fit_function(t_diffs, isi[j], x_plot=[t_diffs[0], t_diffs[-1]],
                                                           method='limited')
        fit_sync_x[j], fit_sync_y[j], _, _, _ = fit_function(t_diffs, sync[j], x_plot=[t_diffs[0], t_diffs[-1]],
                                                           method='exp_decay')
        fit_count_x[j], fit_count_y[j], _, _, _ = fit_function(t_diffs, count[j], x_plot=[t_diffs[0], t_diffs[-1]],
                                                           method='linear')
        fit_dur_x[j], fit_dur_y[j], _, _, _ = fit_function(t_diffs, dur[j], x_plot=[t_diffs[0], t_diffs[-1]],
                                                           method='linear')
        fit_spike_x[j], fit_spike_y[j], _, _, _ = fit_function(t_diffs, spike[j], x_plot=[t_diffs[0], t_diffs[-1]],
                                                           method='power')

    # for j in tqdm(range(len(base_tmax)), desc='Fit: '):
    #     for t in range(len(taus)):
    #         dummy = vr_boots[0][j][:, t]
    #         t_dummy = t_diffs
    #         for kk in range(boots - 1):
    #             dummy = np.append(dummy, vr_boots[kk + 1][j][:, t])
    #             t_dummy = np.append(t_dummy, t_dummy)
    #         fit_vr_x[j][t], fit_vr_y[j][t], _, _, _ = fit_function(t_diffs, vr[j][:, t],
    #                                                                x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                                method='power')
    #     fit_isi_x[j], fit_isi_y[j], _, _, _ = fit_function(t_diffs, isi[j], x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                        method='limited')
    #     fit_sync_x[j], fit_sync_y[j], _, _, _ = fit_function(t_diffs, sync[j], x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                          method='exp_decay')
    #     fit_count_x[j], fit_count_y[j], _, _, _ = fit_function(t_diffs, count[j], x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                            method='linear')
    #     fit_dur_x[j], fit_dur_y[j], _, _, _ = fit_function(t_diffs, dur[j], x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                        method='linear')
    #     fit_spike_x[j], fit_spike_y[j], _, _, _ = fit_function(t_diffs, spike[j], x_plot=[t_diffs[0], t_diffs[-1]],
    #                                                            method='power')

    # Save Data
    # data = {'vr': vr, 'isi': isi, 'sync': sync, 'spike': spike, 'dur': dur, 'count': count, 'base_tmax': base_tmax,
    #         't_diffs': t_diffs, 'taus': taus, 'dt': dt,
    #         'fit_vr': [fit_vr_x, fit_vr_y], 'fit_isi': [fit_isi_x, fit_isi_y],  'fit_sync': [fit_sync_x, fit_sync_y],
    #         'fit_count': [fit_count_x, fit_count_y], 'fit_dur': [fit_dur_x, fit_dur_y],
    #         'fit_spike': [fit_spike_x, fit_spike_y]}

    data = {'vr': vr, 'isi': isi, 'sync': sync, 'spike': spike, 'dur': dur, 'count': count, 'base_tmax': base_tmax,
            't_diffs': t_diffs, 'taus': taus, 'dt': dt,
            'fit_vr': [fit_vr_x, fit_vr_y], 'fit_isi': [fit_isi_x, fit_isi_y], 'fit_sync': [fit_sync_x, fit_sync_y],
            'fit_count': [fit_count_x, fit_count_y], 'fit_dur': [fit_dur_x, fit_dur_y],
            'fit_spike': [fit_spike_x, fit_spike_y], 'boots': boots}

    np.save(path_names[1] + 'poisson/distance_bs_diff.npy', data)
    print('Poisson data saved')

if plot_poisson:
    data = np.load(path_names[1] + 'poisson/distance_bs_diff.npy').item()
    t_diffs = data['t_diffs']
    base_tmax = data['base_tmax']
    taus = data['taus']
    # PLOT
    cc1 = ['0.7', 'LightPink', 'MediumSlateBlue', 'LightGreen', 'PaleGoldenRod']
    cc2 = ['k', 'r', 'b', 'g', 'y']
    mf.plot_settings()
    ax = [[]] * 9
    grid = matplotlib.gridspec.GridSpec(nrows=39, ncols=43)
    fig = plt.figure(figsize=(5.9, 4.9))
    ax[0] = plt.subplot(grid[0:10, 0:10])
    ax[1] = plt.subplot(grid[0:10, 16:26])
    ax[2] = plt.subplot(grid[0:10, 32:42])

    ax[3] = plt.subplot(grid[14:24, 0:10])
    ax[4] = plt.subplot(grid[14:24, 16:26])
    ax[5] = plt.subplot(grid[14:24, 32:42])

    ax[6] = plt.subplot(grid[28:38, 0:10])
    ax[7] = plt.subplot(grid[28:38, 16:26])
    ax[8] = plt.subplot(grid[28:38, 28:42])

    # Subplot caps
    subfig_caps = 12
    label_x_pos = -0.4
    label_y_pos = 1.1
    subfig_caps_labels1 = ['a', 'b', 'c']
    subfig_caps_labels2 = ['d', 'e', 'f', 'f', 'g', 'h']

    # Plot VanRossum ===================================================================================================
    y_limits = [1000, 1000, 2000]
    for i in range(3):
        for k in range(len(base_tmax)):
            # ax[i].plot(t_diffs, data['vr'][k][:, i], marker='', color=cc1[k], alpha=0.8, linestyle='-')
            ax[i].plot(t_diffs, data['vr'][k][:, i], marker='.', color=cc1[k], alpha=0.2, linestyle='', rasterized=True)

        ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels1[i], transform=ax[i].transAxes, size=subfig_caps,
                   color='black')
    for i in range(3):
        for k in range(len(base_tmax)):
            ax[i].plot(data['fit_vr'][0][:, i][k], data['fit_vr'][1][:, i][k], color=cc2[k], lw=1,
                       path_effects=[pe.Stroke(linewidth=1, foreground='w'), pe.Normal()])

        ax[i].text(0.1, 0.9, r'$\tau$ = '+str(taus[i]*1000)+' ms', transform=ax[i].transAxes, size=8, color='black')
        ax[i].set_ylim(0, y_limits[i])
        ax[i].set_yticks([0, 500, 1000])
        ax[i].set_xticks([0, 0.5, 1, 1.5, 2])
    ax[2].set_yticks([0, 1000, 2000])

    profiles = ['isi', 'sync', 'spike', 'dur', 'count']
    y_limits = [1, 1, 1, 2, 250]

    # Plot Profiles ====================================================================================================
    for i in range(3, 8):
        k = i-3
        if k == 2:
            continue
        else:
            for j in range(len(base_tmax)):
                # ax[i].plot(t_diffs, data[profiles[k]][j], marker='', color=cc1[j], alpha=0.8, linestyle='-')
                ax[i].plot(t_diffs, data[profiles[k]][j], marker='.', color=cc1[j], alpha=0.2, linestyle='', rasterized=True)

            ax[i].text(label_x_pos, label_y_pos, subfig_caps_labels2[k], transform=ax[i].transAxes, size=subfig_caps,
                       color='black')
            ax[i].set_ylim(0, y_limits[k])
    # Plot fits
    for i in range(3, 8):
        k = i - 3
        if k == 2:
            continue
        else:
            for j in range(len(base_tmax)):
                ax[i].plot(data['fit_' + profiles[k]][0][j], data['fit_' + profiles[k]][1][j], color=cc2[j], lw=1,
                           path_effects=[pe.Stroke(linewidth=1, foreground='w'), pe.Normal()])
                if i == 7:
                    ax[8].plot(1, 1, color=cc2[j], marker='o', label='Train duration: '+str(base_tmax[j])+' s')
                    ax[8].plot(1, 1, color='w', marker='o', markersize=10)

            ax[i].set_ylim(0, y_limits[k])
            ax[i].set_xticks([0, 0.5, 1, 1.5, 2])

    ax[8].legend(frameon=False, loc=2)
    ax[0].set_ylabel('Van Rossum')
    ax[3].set_ylabel('ISI')
    ax[4].set_ylabel('SYNC')
    # ax[5].set_ylabel('SPIKE')
    ax[6].set_ylabel('DUR')
    ax[7].set_ylabel('COUNT')
    fig.text(0.5, 0.03, 'Difference in spike train duration [s]', ha='center', fontdict=None)

    sns.despine()
    sns.despine(ax=ax[8], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    ax[8].set_xticklabels([])
    ax[8].set_yticklabels([])
    ax[8].set_xticks([])
    ax[8].set_yticks([])

    sns.despine(ax=ax[5], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    ax[5].set_xticklabels([])
    ax[5].set_yticklabels([])
    ax[5].set_xticks([])
    ax[5].set_yticks([])

    # # Estimate for two poisson trains with same rate and duration
    # b = [(rate * base_tmax[i]) for i in range(len(vr))]
    fig.savefig(path_names[2] + 'final/Poisson_DiffInDuration.pdf')
    plt.close(fig)
    print('Poisson Difference in Duration Plot saved')


# JUNKYARD =============================================================================================================
# ======================================================================================================================

# if compute_poisson:
#     boots = 10
#     rate = 100
#     # base_tmax = [0.01, 0.1, 0.5, 1, 2]
#     base_tmax = [0.05, 0.1, 0.5, 1, 2]
#     t_diffs = np.arange(0.001, 2, 0.001)
#     t_diffs = np.insert(t_diffs, 0, 0)
#     taus = [0.001, 0.01, 0.1]
#     dt = 0.001
#     tau = 0.1
#     data = []
#     vr_boots = [[]] * boots
#     isi_boots = [[]] * boots
#     sync_boots = [[]] * boots
#     count_boots = [[]] * boots
#     dur_boots = [[]] * boots
#     spike_boots = [[]] * boots
#     for n in range(boots):
#         vr = np.ndarray((len(base_tmax), len(t_diffs), len(taus)))
#         isi = np.ndarray((len(base_tmax), len(t_diffs)))
#         sync = np.ndarray((len(base_tmax), len(t_diffs)))
#         count = np.ndarray((len(base_tmax), len(t_diffs)))
#         dur = np.ndarray((len(base_tmax), len(t_diffs)))
#         spike = np.ndarray((len(base_tmax), len(t_diffs)))
#
#         pp01, _ = poission_spikes(1, rate, tmax=base_tmax[-1])
#         pp02, _ = poission_spikes(1, rate, tmax=base_tmax[-1] + t_diffs[-1])
#         pp01 = np.array(pp01[0])
#         pp02 = np.array(pp02[0])
#
#         for k in tqdm(range(len(base_tmax)), desc='Poisson Trains: '):
#             for i in range(len(t_diffs)):
#                 # p01, _ = poission_spikes(1, rate, tmax=base_tmax[k])
#                 # p02, _ = poission_spikes(1, rate, tmax=base_tmax[k]+t_diffs[i])
#                 p01 = pp01[pp01 <= base_tmax[k]]
#                 p02 = pp02[pp02 <= base_tmax[k] + t_diffs[i]]
#                 spike_train01 = spk.SpikeTrain(p01, [0, base_tmax[k]+t_diffs[i]])
#                 spike_train02 = spk.SpikeTrain(p02, [0, base_tmax[k]+t_diffs[i]])
#                 for j in range(len(taus)):
#                     pulse01 = mf.fast_e_pulses(np.array(p01), tau=taus[j], dt=dt)
#                     pulse02 = mf.fast_e_pulses(np.array(p02), tau=taus[j], dt=dt)
#                     vr[k][i][j] = mf.vanrossum_distance(pulse01, pulse02, dt=dt, tau=taus[j])
#                 isi[k][i] = spk.isi_distance(spike_train01, spike_train02)
#                 sync[k][i] = spk.spike_sync(spike_train01, spike_train02)
#                 count[k][i] = abs(len(spike_train01.spikes) - len(spike_train02.spikes))
#                 dur[k][i] = abs(spike_train01.spikes[-1] - spike_train02[-1])
#                 spike[k][i] = spk.spike_distance(spike_train01, spike_train02)
#         vr_boots[n] = vr
#         isi_boots[n] = isi
#         sync_boots[n] = sync
#         count_boots[n] = count
#         dur_boots[n] = dur
#         spike_boots[n] = spike
#
#     # FIT
#     fit_vr_x = np.ndarray((len(base_tmax), len(taus), 1000))
#     fit_vr_y = np.ndarray((len(base_tmax), len(taus), 1000))
#     fit_isi_x = np.ndarray((len(base_tmax), 1000))
#     fit_isi_y = np.ndarray((len(base_tmax), 1000))
#     fit_sync_x = np.ndarray((len(base_tmax), 1000))
#     fit_sync_y = np.ndarray((len(base_tmax), 1000))
#     fit_count_x = np.ndarray((len(base_tmax), 1000))
#     fit_count_y = np.ndarray((len(base_tmax), 1000))
#     fit_dur_x = np.ndarray((len(base_tmax), 1000))
#     fit_dur_y = np.ndarray((len(base_tmax), 1000))
#     fit_spike_x = np.ndarray((len(base_tmax), 1000))
#     fit_spike_y = np.ndarray((len(base_tmax), 1000))
#
#     for j in tqdm(range(len(base_tmax)), desc='Fit: '):
#         for t in range(len(taus)):
#             dummy = vr_boots[0][j][:, t]
#             t_dummy1 = np.copy(t_diffs)
#             for kk in range(boots-1):
#                 dummy = np.append(dummy, vr_boots[kk+1][j][:, t])
#                 t_dummy1 = np.append(t_dummy1, t_diffs)
#             fit_vr_x[j][t], fit_vr_y[j][t], _, _, _ = fit_function(t_dummy1, dummy, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                                    method='power')
#         dummy_isi = isi_boots[0][j]
#         dummy_sync = sync_boots[0][j]
#         dummy_count = count_boots[0][j]
#         dummy_dur = dur_boots[0][j]
#         dummy_spike = spike_boots[0][j]
#         t_dummy = np.copy(t_diffs)
#         for kk in range(boots - 1):
#             dummy_isi = np.append(dummy_isi, isi_boots[kk + 1][j])
#             dummy_sync = np.append(dummy_sync, sync_boots[kk + 1][j])
#             dummy_count = np.append(dummy_count, count_boots[kk + 1][j])
#             dummy_dur = np.append(dummy_dur, dur_boots[kk + 1][j])
#             dummy_spike = np.append(dummy_spike, spike_boots[kk + 1][j])
#             t_dummy = np.append(t_dummy, t_diffs)
#
#         fit_isi_x[j], fit_isi_y[j], _, _, _ = fit_function(t_dummy, dummy_isi, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                            method='limited')
#         fit_sync_x[j], fit_sync_y[j], _, _, _ = fit_function(t_dummy, dummy_sync, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                            method='exp_decay')
#         fit_count_x[j], fit_count_y[j], _, _, _ = fit_function(t_dummy, dummy_count, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                            method='linear')
#         fit_dur_x[j], fit_dur_y[j], _, _, _ = fit_function(t_dummy, dummy_dur, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                            method='linear')
#         fit_spike_x[j], fit_spike_y[j], _, _, _ = fit_function(t_dummy, dummy_spike, x_plot=[t_diffs[0], t_diffs[-1]],
#                                                            method='power')
#
#     # for j in tqdm(range(len(base_tmax)), desc='Fit: '):
#     #     for t in range(len(taus)):
#     #         dummy = vr_boots[0][j][:, t]
#     #         t_dummy = t_diffs
#     #         for kk in range(boots - 1):
#     #             dummy = np.append(dummy, vr_boots[kk + 1][j][:, t])
#     #             t_dummy = np.append(t_dummy, t_dummy)
#     #         fit_vr_x[j][t], fit_vr_y[j][t], _, _, _ = fit_function(t_diffs, vr[j][:, t],
#     #                                                                x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                                method='power')
#     #     fit_isi_x[j], fit_isi_y[j], _, _, _ = fit_function(t_diffs, isi[j], x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                        method='limited')
#     #     fit_sync_x[j], fit_sync_y[j], _, _, _ = fit_function(t_diffs, sync[j], x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                          method='exp_decay')
#     #     fit_count_x[j], fit_count_y[j], _, _, _ = fit_function(t_diffs, count[j], x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                            method='linear')
#     #     fit_dur_x[j], fit_dur_y[j], _, _, _ = fit_function(t_diffs, dur[j], x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                        method='linear')
#     #     fit_spike_x[j], fit_spike_y[j], _, _, _ = fit_function(t_diffs, spike[j], x_plot=[t_diffs[0], t_diffs[-1]],
#     #                                                            method='power')
#
#     # Save Data
#     # data = {'vr': vr, 'isi': isi, 'sync': sync, 'spike': spike, 'dur': dur, 'count': count, 'base_tmax': base_tmax,
#     #         't_diffs': t_diffs, 'taus': taus, 'dt': dt,
#     #         'fit_vr': [fit_vr_x, fit_vr_y], 'fit_isi': [fit_isi_x, fit_isi_y],  'fit_sync': [fit_sync_x, fit_sync_y],
#     #         'fit_count': [fit_count_x, fit_count_y], 'fit_dur': [fit_dur_x, fit_dur_y],
#     #         'fit_spike': [fit_spike_x, fit_spike_y]}
#
#     data = {'vr': vr_boots, 'isi': isi_boots, 'sync': sync_boots, 'spike': spike_boots, 'dur': dur_boots, 'count': count_boots, 'base_tmax': base_tmax,
#             't_diffs': t_diffs, 'taus': taus, 'dt': dt,
#             'fit_vr': [fit_vr_x, fit_vr_y], 'fit_isi': [fit_isi_x, fit_isi_y], 'fit_sync': [fit_sync_x, fit_sync_y],
#             'fit_count': [fit_count_x, fit_count_y], 'fit_dur': [fit_dur_x, fit_dur_y],
#             'fit_spike': [fit_spike_x, fit_spike_y], 'boots': boots}
#
#     np.save(path_names[1] + 'poisson/distance_bs_diff.npy', data)
#     print('Poisson data saved')