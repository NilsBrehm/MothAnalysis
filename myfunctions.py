import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
import os
import time
import scipy.io.wavfile as wav
import scipy as scipy
from scipy import signal as sg
from IPython import embed
from shutil import copyfile
# import quickspikes as qs
import itertools as itertools
import pyspike as spk
import pickle
from tqdm import tqdm
# import thunderfish.peakdetection as pk
# from joblib import Parallel,delayed
import csv
# import pycircstat as c_stat
import seaborn as sns
import matplotlib
import warnings
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe
import matplotlib.font_manager
from scipy.stats import norm
from math import exp, sqrt
# import pymuvr

# ----------------------------------------------------------------------------------------------------------------------
# Directories


def get_directories(data_name):
    data_files_path = os.path.join('..', 'figs', data_name, 'DataFiles', '')
    figs_path = os.path.join('..', 'figs', data_name, '')
    nix_path = os.path.join('..', 'mothdata', data_name, data_name)

    path_names = [data_name, data_files_path, figs_path, nix_path]
    return path_names

# ----------------------------------------------------------------------------------------------------------------------
# PEAK DETECTION


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


def plot_cohen(protocol_name, datasets, save_fig):
    plot_settings()

    # d = [[]] * len(datasets)
    d = np.zeros(shape=(17, len(datasets)))
    r = np.zeros(shape=(17, len(datasets)))
    for i in range(len(datasets)):
        data_name = datasets[i]
        path_names = get_directories(data_name=data_name)
        # d[i] = np.load(path_names[1] + protocol_name + '_cohensD.npy')[:17]
        d[:, i] = np.load(path_names[1] + protocol_name + '_cohensD.npy')[:17, 1]
        r[:, i] = np.load(path_names[1] + protocol_name + '_cohensD.npy')[:17, 2]

    gaps = np.flip(np.load(path_names[1] + protocol_name + '_cohensD.npy')[:17, 0], axis=0)
    mean_d = np.flip(np.mean(d, axis=1), axis=0)
    mean_r = np.flip(np.mean(r, axis=1), axis=0)
    max_d = np.flip(np.max(d, axis=1), axis=0)
    min_d = np.flip(np.min(d, axis=1), axis=0)
    max_r = np.flip(np.max(r, axis=1), axis=0)
    min_r = np.flip(np.min(r, axis=1), axis=0)

    # Plot
    marks = ['*-', 'x-', 'd-', 's-', 'v-', '>-', 'h-']
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    fig.set_size_inches(3.9, 1.9)
    fig.subplots_adjust(left=0.15, top=0.98, bottom=0.2, right=0.85, wspace=0.5, hspace=0.2)
    # ax.errorbar(gaps, mean_r, yerr=[mean_r-abs(min_r), max_r-mean_r], color='k', marker='s', label="Pearson's r", markersize=10)
    # ax.fill_between(gaps, min_r, max_r, facecolors='0.25', edgecolor='0.25', alpha=0.5)
    for k in range(r.shape[1]):
        ax.plot(gaps, np.flip(r[:, k], axis=0), marks[k], markersize=1, color='0.6', linewidth=0.5)
    ax.plot(gaps, mean_r, 'k-', label="Pearson's r", markersize=3, linewidth=1)
    ax.set_ylabel("Pearson's r", color='k')
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.2, .2))
    # ax.set_xlabel('Gap [ms]')
    ax.set_xlim(-1, 21)
    ax.set_xticks(np.arange(0, 21, 5))

    ax2 = plt.subplot(1, 2, 2)
    # ax2.errorbar(gaps, mean_d, yerr=[mean_d-abs(min_d), max_d-mean_d], color='0.3', marker='x', linestyle='--', markersize=10)
    # ax2.fill_between(gaps, min_d, max_d, facecolors='0.5', edgecolor='0.5', alpha=0.5)
    for k in range(r.shape[1]):
        ax2.plot(gaps, np.flip(d[:, k], axis=0), marks[k], markersize=1, color='0.6', linewidth=0.5)
    ax2.plot(gaps, mean_d, 'k-', markersize=3, linewidth=1)

    # ax.plot(np.nan, '--x', color='0.3', label="Cohen's d")  # Make an agent in ax for legend

    ax2.set_ylabel("Cohen's d")
    ax2.set_xlim(-1, 21)
    ax2.set_xticks(np.arange(0, 21, 5))

    ax2.set_ylim(0, 11)
    ax2.set_yticks(np.arange(0, 12, 2))

    fig.text(0.5, 0.02, 'Gap [ms]', ha='center', fontdict=None)

    # ax2.spines['right'].set_color('0.3')
    # ax2.tick_params(axis='y', colors='0.3')

    #ax.legend(loc='upper center', shadow=True)
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    if save_fig:
        # Save Plot to HDD
        figname = path_names[1][:8] + protocol_name + '_cohensd.pdf'
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Cohens d plot saved')
    else:
        plt.show()

    return 0


def bootstrap_ci(data, n_boot, level):
    n = len(data)
    idx = np.random.randint(n, size=(n_boot, n))
    resample = data[idx]

    m_data = np.nanmean(data)
    m_boot = np.nanmean(resample, axis=1)

    d_dist = m_boot - m_data

    d1 = np.nanpercentile(d_dist, level+((100-level)/2))
    d2 = np.nanpercentile(d_dist, ((100-level)/2))

    ci = [m_data - d1, m_data - d2]

    return ci


def interval_analysis(path_names, protocol_name, bin_size, save_fig, show, save_data, old, vs_order):
    warnings.catch_warnings()
    warnings.simplefilter("ignore", category=RuntimeWarning)

    # Load Spikes
    plot_settings()
    dataset = path_names[0]
    file_pathname = path_names[1]
    file_name = file_pathname + protocol_name + '_spikes.npy'
    spikes = np.load(file_name).item()

    if protocol_name is 'intervals_mas':
        tag_list = np.arange(0, len(spikes), 1)
        file_name2 = file_pathname + protocol_name + '_meta.npy'
        meta_data = np.load(file_name2).item()
    else:
        tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
        tag_list = np.load(tag_list_path)

    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Get Stimulus Information
    if old:
        stim = [[]] * len(tag_list)
        stim_time = [[]] * len(tag_list)

    # Possion Spikes
    # if protocol_name is not 'intervals_mas':
    if old is False:
        if protocol_name == 'PulseIntervalsRect':
            stim, stim_time, gap, pulse_duration, period = tagtostimulus_gap(path_names, protocol_name)
            tmax = stim_time[0][-1]
            tmin = tmax * 0.2
            pp = np.arange(0.5, 50, 0.5)
        if protocol_name == 'Gap':
            stim, stim_time, gap, pulse_duration, period = tagtostimulus_gap(path_names, protocol_name)
            pp = np.arange(175, 225, 0.5)
            tmax = stim_time[0][-1]
            tmin = tmax * 0.2
        if protocol_name is 'intervals_mas':
            stim = [[]] * len(tag_list)
            stim_time = [[]] * len(tag_list)
            stimT = meta_data[0][0]
            stim_timeT = meta_data[0][1]
            tmax = stim_timeT[-1]
            tmin = tmax * 0.2
            pp = np.arange(0.5, 50, 0.5)
        nsamples = 100
        p_spikes, isi_p = poission_spikes(nsamples, 100, tmax)

        vs_boot, phase_boot, vs_mean_boot, vs_std_boot, vs_percentile_boot, vs_ci_boot = \
            vs_range(p_spikes, pp/1000, tmin=tmin, n_ci=0, order=vs_order)

    vector_strength = np.zeros(shape=(len(tag_list), 8))
    aa = 0
    cors = np.zeros(shape=(len(tag_list), 5))
    # cohen_d = np.zeros(shape=(len(tag_list), 3))
    # Loop trough all tags in tag_list
    for i in tqdm(range(len(tag_list)), desc='Interval Analysis'):

        # Try to get Spike Times
        try:
            spike_times = spikes[tag_list[i]]
        except:
            if aa == 0:
                stop_id = i
            aa = 2
            continue

        # Convert stimulus parameters into floats
        if protocol_name is 'intervals_mas':
            stim[i] = meta_data[i][0]
            stim_time[i] = meta_data[i][1]
            gaps = np.round(np.float(meta_data[i][2]) * 1000)
            pd = np.float(meta_data[i][3]) * 1000
            period_num = pd + gaps
        else:
            period_num = int(float(period[i]))
            pd = int(float(pulse_duration[i]))
            gaps = period_num - pd

        # For Old MAS the Poisson Spikes must be computed for each stimulus
        if protocol_name is 'intervals_mas' and old is True:
            stim[i] = meta_data[i][0]
            stim_time[i] = meta_data[i][1]
            tmax = stim_time[i][-1]
            tmin = tmax * 0.2
            nsamples = 100
            p_spikes, isi_p = poission_spikes(nsamples, 100, tmax)
            pp = np.arange(0.5, 50, 0.5)

            vs_boot, phase_boot, vs_mean_boot, vs_std_boot, vs_percentile_boot, vs_ci_boot = \
                vs_range(p_spikes, pp / 1000, tmin=tmin, n_ci=0, order=vs_order)

        # VS Range
        vs, phase, vs_mean, vs_std, vs_percentile, vs_ci = vs_range(spike_times, pp / 1000, tmin=tmin, n_ci=100, order=vs_order)

        # Poisson Boot
        # Project spike times onto circle and get the phase (angle)
        if protocol_name == 'PulseIntervalsRect':
            t_period = float(period[i]) / 1000
            idx = pp == period_num
            if gaps < 1:
                # print('gap 0')
                idx = pp == 40

        if protocol_name == 'Gap':  # Use gap as period for VS
            # t_period = float(gap[i]) / 1000
            t_period = float(period[i]) / 1000
            idx = pp == period_num
        if protocol_name is 'intervals_mas':
            t_period = float(gaps) / 1000
            idx = pp == int(gaps)

        # Project spike times onto circle and get the phase (angle)
        # v = np.exp(2j * np.pi * spike_times/t_period)
        vector_boot = np.exp(np.dot(2j * np.pi / t_period, np.concatenate(p_spikes)))
        vector_phase_boot = np.angle(vector_boot)
        vector = np.exp(np.dot(2j * np.pi / t_period, np.concatenate(spike_times)))
        vector_phase = np.angle(vector)
        mu = np.mean(vector)

        # Rayleigh Test
        # p_boot_rayleigh, z_boot_rayleigh = c_stat.rayleigh(vector_phase_boot)
        # n = len(p_spikes)
        # rr_boot = n * vs_mean_boot[idx][0]
        # p_boot_rayleigh = np.exp(-n * rr_boot ** 2)
        # p_rayleigh, z_rayleigh = c_stat.rayleigh(vector_phase)
        # n = len(spike_times)
        # rr = n * vs_mean[idx][0]
        # p_rayleigh = np.exp(-n*rr**2)
        # # V-Test
        # # p_vtest, z_vtest = c_stat.vtest(vector_phase, np.angle(mu))
        #
        # # print('poisson: p=' + str(p_boot_rayleigh) + ', z=' + str(rr_boot))
        # if p_rayleigh < 0.001:
        #     print(str(gaps) + ' ms: p<' + str(0.001) + '*, z=' + str(rr))
        # else:
        #     print(str(gaps) + ' ms: p=' + str(p_rayleigh) + ', z=' + str(rr))
        # print('--------------')

        # Cohens d and pearsons r
        # cohen_d[i, 0] = gaps
        # sd_pooled = np.sqrt((vs_std[idx][0]**2 + vs_std_boot[idx][0]**2)/2)
        # cohen_d[i, 1] = (vs_mean[idx][0]-vs_mean_boot[idx][0]) / sd_pooled
        # cohen_d[i, 2] = cohen_d[i, 1] / np.sqrt(cohen_d[i, 1]**2 + 4)

        # CORRELATION ==================================================================================================
        # Compute Convolved Firing Rate
        dt = 1 / fs
        sigma = bin_size  # in seconds
        rate_conv_time, rate_conv, rate_conv_std, conv_overall_frate, conv_overall_frate_std = convolution_rate(
            spike_times, tmax, dt, sigma, method='mean')

        # Distances Profiles
        x_limit = stim_time[i][-1]
        trains = [[]] * len(spike_times)
        edges = [0, 1]
        for q in range(len(spike_times)):
            trains[q] = spk.SpikeTrain(spike_times[q], edges)
        isi_profile = spk.isi_profile(trains)
        spike_profile = spk.spike_profile(trains)
        sync_profile = spk.spike_sync_profile(trains)
        isi_x, isi_y = isi_profile.get_plottable_data()
        spike_x, spike_y = spike_profile.get_plottable_data()
        sync_x, sync_y = sync_profile.get_plottable_data()

        # Running Average
        N = 10
        sync_ra = np.convolve(sync_y, np.ones((N,)) / N, mode='valid')
        t_sync_ra = np.linspace(0, x_limit, len(sync_ra))

        # Correlation
        # duty_cycle = pd / period_num
        # rect1 = scipy.signal.square(2 * np.pi * (1 / (period_num / 1000)) * rate_conv_time, duty=duty_cycle)
        sin1 = np.sin(rate_conv_time * 2 * np.pi * (1 / (period_num / 1000)))
        sin2 = np.sin(t_sync_ra * 2 * np.pi * (1 / (period_num / 1000)))

        # normalize
        x1 = (rate_conv - np.mean(rate_conv)) / (np.std(rate_conv))
        x2 = (sync_ra - np.mean(sync_ra)) / (np.std(sync_ra))
        # x13 = (rect1 - np.mean(rect1)) / (np.std(rect1))
        x13 = (sin1 - np.mean(sin1)) / (np.std(sin1))
        x23 = (sin2 - np.mean(sin2)) / (np.std(sin2))
        fs1 = len(x1) / stim_time[i][-1]
        fs2 = len(x2) / stim_time[i][-1]
        lim1 = int(fs1 * (stim_time[i][-1] * 0.2))
        lim2 = int(fs2 * (stim_time[i][-1] * 0.2))
        r1 = scipy.signal.correlate(x1[lim1:], x13[lim1:]) / len(x1[lim1:])
        r2 = scipy.signal.correlate(x2[lim2:], x23[lim2:]) / len(x2[lim2:])
        # r1 = abs(r1)
        # r2 = abs(r2)
        lag1 = (np.where(r1 == np.max(r1))[0][0] - (len(r1)/2)) / fs1
        lag2 = (np.where(r2 == np.max(r2))[0][0] - (len(r2)/2)) / fs2
        # gaps | corr rate | corr sync | lag rate | lag sync
        cors[i, :] = [gaps, np.max(r1), np.max(r2), lag1, lag2]

        # # print(str(period_num) + ': rate: ' + str(np.max(r1)) + ', sync: ' + str(np.max(r1)))
        # plt.subplot(2, 1, 1)
        # plt.plot(x1, 'k')
        # plt.plot(x13, 'r')
        # plt.title(str(gaps) + ': ' + str(np.max(r1)))
        # # plt.plot(r1)
        # # plt.axvline(len(r1)/2, color='r')
        # plt.subplot(2, 1, 2)
        # plt.plot(x2, 'k')
        # plt.plot(x23, 'r')
        # plt.title(str(gaps) + ': ' + str(np.max(r2)))
        # # plt.plot(r2)
        # # plt.axvline(len(r2) / 2, color='r')
        # plt.tight_layout()
        # plt.show()

        # Put Data together
        vs = [[]] * 7
        vs[0] = vs_mean
        vs[1] = vs_ci
        vs[2] = vector_phase
        vs[3] = mu
        vs[4] = np.nan
        vs[5] = pp
        vs[6] = period_num
        if gaps < 1:
            vs[6] = 40

        vs_boot = [[]] * 4
        vs_boot[0] = vs_mean_boot
        vs_boot[1] = vs_std_boot
        vs_boot[2] = vs_percentile_boot
        vs_boot[3] = vector_phase_boot

        if vs_order == 2:
            try:
                vector_strength[i, :] = [period_num, pd, gaps, vs_mean[idx][0], vs_ci[idx, 0], vs_ci[idx, 1],
                                         vs_mean_boot[idx][0], vs_percentile_boot[idx][0]]
            except:
                embed()
                exit()
        if vs_order == 1:
            vector_strength[i, :] = [period_num, pd, gaps, vs_mean[idx][0], vs_ci, vs_ci,
                                     vs_mean_boot[idx][0], vs_percentile_boot]

        # Now Plot it
        uu = gaps == np.array([20, 15, 10, 5, 4, 3, 2, 1, 0])
        # uu = gaps == np.array([20, 10])
        if show[0] and uu.any():
            # Adapt bin size to half the gap size
            if gaps < 0:
                bin_size = gaps / 2000
            else:
                bin_size = 10 / 2000

            plot_gaps(spike_times, stim_time[i], stim[i], bin_size, p_spikes, isi_p, vs, vs_boot, mark_intervals=False)
            if save_fig:
                # Save Plot to HDD
                figname = path_names[2] + 'id_' + str(i) + '_' + protocol_name + '_pd_' + str(pd) + '_gap_' + str(gaps) + '_VSorder_' + str(vs_order) +'.pdf'
                fig = plt.gcf()
                # fig.set_size_inches(5.9, 5.9)
                # fig.subplots_adjust(left=0.15, top=0.9, bottom=0.1, right=0.9, wspace=0.005, hspace=0.8)
                # fig.savefig(figname, bbox_inches='tight', dpi=300)
                fig.savefig(figname)
                plt.close(fig)
            # else:
            #     plt.show()

    fig, ax = plt.subplots()
    if show[1]:
        if aa == 0:
            plot_vs_gap(vector_strength, ax, protocol_name)
            print('VS Data Cutting Mode: ' + str(1))
        elif aa == 2 and protocol_name is 'intervals_mas':
            plot_vs_gap(vector_strength[1:stop_id], ax, protocol_name)
            print('VS Data Cutting Mode: ' + str(2))
        elif aa == 2 and protocol_name is not 'intervals_mas':
            plot_vs_gap(vector_strength[0:34], ax, protocol_name)
            print('VS Data Cutting Mode: ' + str(3))

    sns.despine(fig=fig)
    fig.set_size_inches(2.9, 1.9)
    fig.subplots_adjust(left=0.2, top=0.98, bottom=0.2, right=0.98, wspace=0.4, hspace=0.4)

    if save_fig:
        # Save Plot to HDD
        figname = path_names[2] + protocol_name + '_VS_vs_gaps_VSorder_' + str(vs_order) +'.pdf'
        fig = plt.gcf()
        # fig.set_size_inches(5, 5)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
    # else:
    #     plt.show()

    # Save Data to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_vs.npy'
        np.save(file_name, vector_strength)
        np.save(file_pathname + protocol_name + '_corr.npy', cors)
        print('VS saved (protocol: ' + protocol_name + ')')
    return 0


def poisson_vs_duration(ts, vs_tmax, rates, mode):
    def func(x, a, c, d):
        # return a * np.exp(-d * x) + c
        # return a * np.log(d * x) + c
        # return a * k**(-d * x) + c
        return a * np.exp(-d * x) + c

    def func2(x, d, c):
        # return np.exp(-d * x) + c
        return -d * x + c
        # return a * x ** -d + c

    if np.isnan(vs_tmax).any():
        print('Warning: Found NaNs')

    plot_settings()
    fig, ax = plt.subplots()
    cc = ['black', 'blue', 'red', 'green', 'yellow', 'magenta']
    init_vals = [0.5, 0.5, 0.01]  # for [a, c, d]
    for i in range(vs_tmax.shape[0]):
        vs = np.delete(vs_tmax[i], np.where(np.isnan(vs_tmax[i])))
        ts_no_nan = np.delete(ts, np.where(np.isnan(vs_tmax[i])))
        id_fit2 = ts_no_nan >= ts[-1]*0.5
        id_fit = ts_no_nan <= ts[-1]

        popt, pcov = curve_fit(func, ts_no_nan[id_fit], vs[id_fit], maxfev=10000)
        popt2, pcov2 = curve_fit(func2, ts_no_nan[id_fit2], vs[id_fit2], maxfev=1000)

        if mode is 'rates':
            ax.plot(ts_no_nan, vs, '.', label=str(rates[i]) + ' Hz', color=cc[i], markersize=1, alpha=0.4)
            ax.plot(ts_no_nan[ts_no_nan <= ts[-1]*0.5], func(ts_no_nan[ts_no_nan <= ts[-1]*0.5], *popt), '-', color=cc[i], linewidth=0.5)
            ax.plot(ts_no_nan[id_fit], func(ts_no_nan[id_fit], *popt), '-', color=cc[i], linewidth=0.5, alpha=0.25)
            ax.plot(ts_no_nan[id_fit2], func2(ts_no_nan[id_fit2], *popt2), '--', color=cc[i], linewidth=0.5)

        if mode is 'nsamples':
            cc = ['0.9', '0.6', '0.2']
            symbols = ['s', 'd', 'o']
            ax.plot(ts_no_nan, vs, 'o', label='n = ' + str(rates[i]), markerfacecolor=cc[i], markeredgecolor='black', markersize=1.5, markeredgewidth=0.2)
            # ax.scatter(ts_no_nan, vs,s=1, color=cc[i], label='n = ' + str(rates[i]))

    # if mode is 'nsamples':
    #     ax.plot(ts_no_nan[ts_no_nan <= ts[-1] * 0.5], func(ts_no_nan[ts_no_nan <= ts[-1] * 0.5], *popt), '-',
    #             color='k', linewidth=0.5)
    #     ax.plot(ts_no_nan[id_fit], func(ts_no_nan[id_fit], *popt), '-', color='k', linewidth=0.5, alpha=0.25)
    #     ax.plot(ts_no_nan[id_fit2], func2(ts_no_nan[id_fit2], *popt2), '--', color='k', linewidth=0.5)
    ax.set_xlabel('Spike Train Duration [ms]')
    ax.set_ylabel('Mean Vector Strength')
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend()
    sns.despine()
    return 0


def fit_function(x_fit, data):
    def func(xx, bottom, top, V50, Slope):
        # return a * np.exp(-d * x1) + c
        # return bottom + ((top-bottom)/(1+np.exp((V50-xx)/Slope)))
        return bottom + ((top-bottom)/(1+np.exp(-Slope*(xx-V50))))

    # Check for Nans
    if np.isnan(data).all():
        x, y = [np.nan] * 2
        popt = [np.nan] * 4
        perr = [np.inf] * 4
        return x, y, popt, perr
    # Remove NaNs
    idx = ~np.isnan(data)
    data = data[idx]
    x_fit = x_fit[idx]
    p0 = [0, np.max(data), np.max(x_fit)/2, 1]
    bounds = (0, [np.max(data)/2+10, np.max(data)*2+10, np.max(x_fit), np.inf])
    # print('p0:')
    # print(p0)
    # print('bounds:')
    # print(bounds)
    # print('-----------------')
    popt, pcov = curve_fit(func, x_fit, data, p0=p0, maxfev=10000,  bounds=bounds)
    x = np.linspace(np.min(x_fit), np.max(x_fit), 1000)
    y = func(x, *popt)
    y0 = func(popt[-2], *popt)
    perr = np.sqrt(np.diag(pcov))
    return x, y, popt, perr, y0


def poisson_vs_duration2(ts, vs_rates, vs_samples, rates, nsamples):
    def func(x, a, c, d):
        # return a * np.exp(-d * x) + c
        # return a * np.log(d * x) + c
        # return a * k**(-d * x) + c
        return a * np.exp(-d * x) + c

    def func2(x, a,  d, c):
        # return np.exp(-d * x) + c
        # return -d * x + c
        return a * x ** -d + c
        # return a * np.exp(-d * x) + c

    if np.isnan(vs_rates).any() or np.isnan(vs_samples).any():
        print('Warning: Found NaNs')

    plot_settings()
    fig_size = (1, 2)
    fig = plt.figure()
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.5, hspace=0.4)
    samples_ax = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=1)
    rates_ax = plt.subplot2grid(fig_size, (0, 1), rowspan=1, colspan=1)

    samples_ax.set_ylabel('Mean vector strength')
    # samples_ax.set_xlabel('Spike Train Duration [ms]')
    samples_ax.set_ylim(0, 1)
    samples_ax.set_xlim(0, 1)
    samples_ax.set_yticks(np.arange(0, 1.1, 0.2))
    samples_ax.set_xticks(np.arange(0, 1.1, 0.2))
    cc = ['0', '0.4', '0.85']
    for k in range(vs_samples.shape[0]):
        vs = np.delete(vs_samples[k], np.where(np.isnan(vs_samples[k])))
        ts_no_nan = np.delete(ts, np.where(np.isnan(vs_samples[k])))
        samples_ax.plot(ts_no_nan, vs, '-', label=str(nsamples[k]) + ' trial(s)', color=cc[k])
    samples_ax.legend(frameon=False)

    cc = ['0', '0.25', '0.4', '0.8']
    for i in range(vs_rates.shape[0]):
        vs = np.delete(vs_rates[i], np.where(np.isnan(vs_rates[i])))
        ts_no_nan = np.delete(ts, np.where(np.isnan(vs_rates[i])))
        # id_fit2 = ts_no_nan >= ts[-1]*0.5
        # id_fit = ts_no_nan <= ts[-1]
        #
        # popt, pcov = curve_fit(func, ts_no_nan[id_fit], vs[id_fit], maxfev=10000)
        # popt2, pcov2 = curve_fit(func2, ts_no_nan, vs, maxfev=1000)
        rates_ax.plot(ts_no_nan, vs, '-', label=str(rates[i]) + ' Hz', color=cc[i])

    # rates_ax.set_xlabel('Spike Train Duration [ms]')
    rates_ax.set_ylim(0, 1)
    rates_ax.set_xlim(0, 1)
    rates_ax.set_yticks(np.arange(0, 1.1, 0.2))
    rates_ax.set_xticks(np.arange(0, 1.1, 0.2))
    rates_ax.legend(frameon=False)
    subfig_caps = 12
    label_x_pos = -0.3
    label_y_pos = 1
    samples_ax.text(label_x_pos, label_y_pos, 'a', transform=samples_ax.transAxes, size=subfig_caps)
    rates_ax.text(label_x_pos, label_y_pos, 'b', transform=rates_ax.transAxes, size=subfig_caps)

    fig.text(0.55, 0.02, 'Spike train duration [s]', ha='center', fontdict=None)
    sns.despine()
    return 0


def plot_vs_gap(vs, ax, protocol_name):
    # vs: period, pulse duration, gap, vs mean, vs std, vs boot mean, vs boot percentile
    plot_settings()
    # Sort Input after gaps
    sort_id = np.argsort(vs[:, 2])
    vs = vs[sort_id]

    pd = vs[:, 1]
    if protocol_name == 'PulseIntervalsRect':
        pd_idx_5 = pd == 5
        pd_idx_10 = pd == 10
        label_10 = '10 ms'
        label_5 = '5 ms'
    if protocol_name is 'intervals_mas':
        pd_idx_5 = pd == 0.4
        pd_idx_10 = pd == 0.1
        label_10 = '0.1 ms'
        label_5 = '0.4 ms'
    if protocol_name is 'Gap':
        pd_idx_10 = np.array([True] * len(pd))
        pd_idx_5 = np.array([False])
        label_10 = '10 ms'
        label_5 = '5 ms'
    gaps = vs[:, 2]
    vs_mean = vs[:, 3]
    vs_ci_low = vs[:, 4]
    vs_ci_up = vs[:, 5]
    vs_boot = vs[:, 6]
    vs_boot_percentile = vs[:, 7]

    if pd_idx_10.any():
        low = vs_mean[pd_idx_10] - vs_ci_low[pd_idx_10]
        up = vs_ci_up[pd_idx_10] - vs_mean[pd_idx_10]
        ax.errorbar(gaps[pd_idx_10], vs_mean[pd_idx_10], yerr=[low, up], marker='s', color='k', label=label_10)
    if pd_idx_5.any():
        low = vs_mean[pd_idx_5] - vs_ci_low[pd_idx_5]
        up = vs_ci_up[pd_idx_5] - vs_mean[pd_idx_5]
        ax.errorbar(gaps[pd_idx_5], vs_mean[pd_idx_5], yerr=[low, up], marker='d', color='0.4', label=label_5)
    if pd_idx_10.any():
        ax.plot(gaps, vs_boot, 'k--', label='poisson spikes', linewidth=0.5)
        # ax.plot(gaps, vs_boot_percentile, 'k:', label='95 % perc', linewidth=0.5)
        ax.fill_between(gaps, np.zeros(len(gaps)), vs_boot_percentile, label='95 % perc', facecolors='0.5', alpha=0.5)
    ax.legend(loc='best', shadow=False, frameon=False)
    ax.set_ylabel('Vector Strength')
    ax.set_xlabel('Gaps [ms]')
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, np.max(gaps)+1, 2))
    ax.set_xlim(-1, np.max(gaps)+1)

    return 0


def plot_spike_detection_gaps_backup(x, spike_times, spike_times_valley, marked, spike_size, mph_percent, snippets,
                              snippets_removed, th, window, tag_list, stim_time, stim):
    # Find 10 ms Intervals
    fs = 100 * 1000
    isi = np.diff(spike_times)
    a = isi > 0.008*fs
    b = isi < 0.015*fs
    idx = a == b
    idx1 = np.append(idx, False)
    idx2 = np.insert(idx, 0, False)
    marked_spikes = spike_times[idx1]  # First spike in interval 10 ms
    marked_spikes2 = spike_times[idx2]  # Second spike in interval 10 ms

    x_limit = stim_time[-1]+0.2
    fig = plt.figure(figsize=(12, 8))
    fig_size = (3, 2)
    spike_shapes = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=1)
    spike_width = plt.subplot2grid(fig_size, (0, 1), rowspan=1, colspan=1)
    volt_trace = plt.subplot2grid(fig_size, (1, 0), rowspan=1, colspan=2)
    stim_trace = plt.subplot2grid(fig_size, (2, 0), rowspan=1, colspan=2, sharex=volt_trace)

    # Plot Spike Shapes ================================================================================================
    t_snippets = np.arange(-100 / fs, 100 / fs, 1 / fs) * 1000
    for s in range(len(snippets)):
        spike_shapes.plot(t_snippets, snippets[s], 'k')
    for s in range(len(snippets_removed)):
        spike_shapes.plot(t_snippets, snippets_removed[s], 'r')

    spike_shapes.set_xlabel('Time [ms]')
    spike_shapes.set_ylabel('Voltage [uV]')

    # Plot Spike Parameters ============================================================================================
    # Width vs Size
    spike_width.plot(spike_size[:, 3] * 1000, spike_size[:, 2], 'ko')
    spike_width.set_xlabel('Spike Width [ms]')
    spike_width.set_ylabel('Spike Size [uV]')

    # Width vs Height
    ax2 = spike_width.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(spike_size[:, 3] * 1000, spike_size[:, 1], 'gx')
    ax2.plot([np.min(spike_size[:, 3] * 1000), np.max(spike_size[:, 3] * 1000)],
             [np.max(spike_size[:, 1]) * mph_percent, np.max(spike_size[:, 1]) * mph_percent], 'g--')
    ax2.set_ylabel('Spike Height[uV]', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Plot Voltage and detected spikes =================================================================================
    plot_detected_spikes(x, spike_times, spike_times_valley, marked, th, window, tag_list, volt_trace)
    # volt_trace.set_xticks([])
    for k in range(len(marked_spikes)):
        # volt_trace.plot([marked_spikes[k], marked_spikes[k]+0.01], [np.max(x), np.max(x)], 'gx--')
        volt_trace.plot([marked_spikes[k]/fs, marked_spikes2[k]/fs], [x[marked_spikes[k]], x[marked_spikes2[k]]], 'gx--')
    # volt_trace.plot(marked_spikes, np.zeros(len(marked_spikes))+np.max(x), 'go')
    volt_trace.set_xlim(0, x_limit)

    # Plot Stimulus ====================================================================================================
    stim_trace.plot(stim_time, stim, 'k')
    for k in range(len(marked_spikes)):
        stim_trace.plot([marked_spikes[k] / fs, marked_spikes2[k] / fs], [1, 1], 'g--', linewidth=4)
    stim_trace.set_xlim(0, x_limit)
    fig.tight_layout()
    plt.show()


def plot_spike_detection_gaps(x, spike_times, spike_times_valley, marked, spike_size, mph_percent, snippets,
                              snippets_removed, th, window, tag_list, stim_time, stim):

    plot_settings()
    detect_on = True
    x_limit = (stim_time[-1] + 0.1) * 1000
    fig = plt.figure(figsize=(5.9, 3.9))
    fig_size = (23, 1)
    stim_trace = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=1)
    volt_trace = plt.subplot2grid(fig_size, (1, 0), rowspan=10, colspan=1)
    close_up = plt.subplot2grid(fig_size, (13, 0), rowspan=22, colspan=1)

    # Plot Stimulus ====================================================================================================
    stim_trace.plot(stim_time*1000, stim, 'k')
    stim_trace.set_xlim(-100, x_limit)
    sns.despine(ax=stim_trace,  top=True, right=True, left=True, bottom=True)
    stim_trace.set_xticks([])
    stim_trace.set_yticks([])

    # Plot Voltage and detected spikes =================================================================================
    # plt.figure()
    fs = 100 * 1000
    t = np.arange(-0.1, (len(x) / fs)-0.1, 1 / fs)
    volt_trace.plot(t*1000, x, 'k')
    if detect_on:
        volt_trace.plot(((marked/fs)-0.1)*1000, x[marked], 'bx', markersize=4)
        volt_trace.plot(((spike_times/fs)-0.1)*1000, x[spike_times], 'r.', markersize=4)
    volt_trace.set_ylabel('Voltage [uV]')
    volt_trace.set_xlim(-100, x_limit)
    volt_trace.set_ylim(-300, 300)
    volt_trace.set_yticks([-300, -150, 0, 150, 300])
    sns.despine(ax=volt_trace, top=True, right=True, left=False, bottom=False,)

    start = int(0.1 * fs)
    stop = int(0.2 * fs)
    close_up.plot((t[start:stop])*1000, x[start:stop], 'k')
    close_up.plot((t[start:stop])*1000, x[start:stop], 'k')
    close_up.plot(((spike_times / fs) - 0.1) * 1000, np.zeros(len(spike_times))+150, 'r.', markersize=4)
    close_up.plot(((marked / fs) - 0.1) * 1000, np.zeros(len(marked))+250, 'bx', markersize=4)

    sns.despine(ax=close_up, top=True, right=True, left=False, bottom=False,)
    close_up.set_ylabel('Voltage [uV]')
    close_up.set_xlabel('Time [ms]')
    close_up.set_xticks(np.arange(0, 101, 10))
    close_up.set_xlim(0, 100)
    close_up.set_ylim(-300, 300)
    close_up.set_yticks([-300, -150, 0, 150, 300])

    # plt.show()
    if detect_on:
        fig.savefig('/media/brehm/Data/MasterMoth/figs/spikes_rect_detected_ ' + tag_list + '.pdf')
    else:
        fig.savefig('/media/brehm/Data/MasterMoth/figs/spikes_rect_ '+  tag_list +'.pdf')
    plt.close()
    exit()


def plot_gaps(spike_times, stim_time, stim, bin_size, p_spikes, isi_p, vs, vs_boot, mark_intervals):
    """Plot Raster Plot, PSTH and Spike Distances over all trials

       Notes
       ----------
       This function plots one spike train out of all trials and a Raster Plot, PSTH, Spike Train Distances and the
       Stimulus

       Parameters
       ----------
       spike_times: Spike Times
       info: Information about stimulus
       stim_time: Stimulus time
       stim: Stimulus
       bin_size: Bin Size for PSTH
       p_spikes: Poisson Spikes
       isi_p: Poisson Inter Spike Intervals
       vs: Vector Strength data (list)
       vs_boot: Poisson Vector Strenght data (list)

       Returns
       -------
       Plot
       """
    plot_settings()
    # Voltage Sampling Rate [Hz]
    fs = 100*1000
    if np.isnan(vs[1]).all():
        vs_order = 1
    else:
        vs_order = 2

    # Compute PSTH
    tmin = 0.005
    tmax = stim_time[-1]
    bins = int(stim_time[-1] / bin_size)
    hh = [[]] * len(spike_times)
    for h in range(len(spike_times)):
        hh[h], bin_edges = np.histogram(spike_times[h], bins)
        hh[h] = hh[h] / bin_size
    frate = np.mean(hh, axis=0)
    frate_std = np.std(hh, axis=0)
    overall_frate = np.mean(hh, axis=1)
    overall_frate_std = np.std(overall_frate)
    overall_frate = np.mean(overall_frate)
    mean_rate = overall_frate
    mean_period = (1/overall_frate) * 1000

    # Compute Convolved Firing Rate
    dt = 1 / fs
    sigma = bin_size  # in seconds
    rate_conv_time, rate_conv, rate_conv_std, conv_overall_frate, conv_overall_frate_std = convolution_rate(spike_times, tmax, dt, sigma, method='mean')
    # Ignore the first 20 % of trial for mean firing rate
    # limits = np.int((len(rate_conv)/fs)*0.2 * fs)
    # conv_overall_frate = np.mean(rate_conv[limits:])
    # conv_overall_frate_std = np.std(rate_conv[limits:])

    # Get Vector Strength Data
    vs_mean = vs[0]
    vs_ci = vs[1]
    vector_phase = vs[2]
    mu = vs[3]
    peaks = vs[4]
    pp = vs[5]
    period = np.floor(vs[6])

    vs_mean_boot = vs_boot[0]
    vs_std_boot = vs_boot[1]
    vs_percentile_boot = vs_boot[2]
    vector_phase_boot = vs_boot[3]

    # Prepare Axes
    subfig_caps = 12
    x_limit = stim_time[-1]
    fig = plt.figure()
    # fig.set_size_inches(3.9, 3.1)
    # fig.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95, wspace=0.15, hspace=0.5)

    fig_size = (4, 3)
    fig.set_size_inches(5.9, 5.9)
    fig.subplots_adjust(left=0.15, top=0.9, bottom=0.1, right=0.9, wspace=0.6, hspace=0.8)
    # Creat Grid
    grid = matplotlib.gridspec.GridSpec(nrows=41, ncols=3)
    stim1 = plt.subplot(grid[0:2, :])
    raster = plt.subplot(grid[2:11, :])
    psth_ax = plt.subplot(grid[14:24, :])
    isi_ax = plt.subplot(grid[30:40, 0])
    vs_hist_ax = plt.subplot(grid[30:40, 1])
    vs_ax = plt.subplot(grid[30:40, 2])


    # isi_ax = plt.subplot2grid(fig_size, (3, 0), rowspan=1, colspan=1)
    # vs_ax = plt.subplot2grid(fig_size, (3, 2), rowspan=1, colspan=1)
    # vs_hist_ax = plt.subplot2grid(fig_size, (3, 1), rowspan=1, colspan=1)
    # raster = plt.subplot2grid(fig_size, (1, 0), rowspan=1, colspan=3)
    # psth_ax = plt.subplot2grid(fig_size, (2, 0), rowspan=1, colspan=3, sharex=raster)
    # distance = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=3, sharex=raster)

    label_x_pos1 = -0.60
    label_x_pos2 = label_x_pos1 / 4
    label_y_pos = 1

    yaxis_pos1 = -0.08
    yaxis_pos2 = yaxis_pos1 * 4

    # ISI Histogram ====================================================================================================
    bs = 1  # in ms
    plot_isi_histogram(spike_times, p_spikes, bs, x_limit=50, steps=5, info=[''],
                       intervals=isi_p, tmin=tmin, plot_title=False, ax=isi_ax)
    # isi_ax.legend(loc='upper right', shadow=False, frameon=False)
    isi_ax.set_xlim(0, 30)
    isi_ax.set_xticks(np.arange(0, 30+5, 5))
    isi_ax.xaxis.set_label_coords(0.5, -0.3)
    isi_ax.yaxis.set_label_coords(yaxis_pos2, 0.5)
    sns.despine(ax=isi_ax)
    isi_ax.legend(loc='best', shadow=False, frameon=False)


    # VS vs period =====================================================================================================
    # vs_ax.plot(pp, vs_std_boot + vs_mean_boot, 'g--')
    # if peaks.any():
    #     vs_ax.plot(pp[peaks], vs_mean[peaks], 'bo')
    # vs_ax.plot([float(period), float(period)], [0, 1], 'bx:', linewidth=0.3)
    vs_ax.plot(pp, vs_mean, 'k', label='Data')
    vs_ax.plot(pp, vs_mean_boot, '-', label='Poisson', linewidth=0.5, color='1')
    if vs_order == 2:
        # vs_ax.plot(pp, vs_percentile_boot, ':', linewidth=0.5, color='0.5')
        vs_ax.fill_between(pp, [0]*len(vs_percentile_boot), vs_percentile_boot, facecolor='0.5', alpha=0.75)
        # vs_ax.fill_between(pp, vs_ci[:, 0], vs_ci[:, 1], facecolor='k', alpha=0.5)
    if period < 39:
        vs_ax.text(period - 0.5, 0.95, r'$s$', size=5)
        vs_ax.arrow(period, 0.9, 0, -0.85, head_width=0.75, head_length=0.025, fc='k', ec='k', head_starts_at_zero=False,
                    alpha=0.75, linewidth=0.75)
        vs_ax.arrow(period / 2, 0.9, 0, -0.85, head_width=0.75, head_length=0.025, fc='k', ec='k',
                    head_starts_at_zero=False,
                    alpha=0.75, linewidth=0.75)
        vs_ax.arrow(period / 4, 0.9, 0, -0.85, head_width=0.75, head_length=0.025, fc='k', ec='k',
                    head_starts_at_zero=False,
                    alpha=0.75, linewidth=0.75)
        vs_ax.text(period / 2 - 0.5, 0.95, r'$\frac{s}{2}$', size=5)
        vs_ax.text(period / 4 - 0.5, 0.95, r'$\frac{s}{4}$', size=5)
    vs_ax.arrow(mean_period, 0.9, 0, -0.85, head_width=0.75, head_length=0.025, fc='k', ec='k',
                head_starts_at_zero=False,
                alpha=0.75, linewidth=0.75)
    vs_ax.text(mean_period-0.5, 0.95, r'$f$', size=5)

    vs_ax.set_xlabel('Period [ms]')
    vs_ax.set_ylabel('Mean VS')
    vs_ax.set_xlim(-0.2, np.max(pp))
    vs_ax.set_xticks(np.arange(0, np.max(pp)+10, 10))
    if np.min(pp) > 10:
        vs_ax.set_xlim(np.min(pp), np.max(pp))
        vs_ax.set_xticks(np.arange(np.min(pp), np.max(pp)+10, 10))
    vs_ax.set_ylim(0, 1)
    vs_ax.set_yticks(np.arange(0, 1.1, 0.5))
    # vs_ax.legend(loc='best', shadow=False, frameon=False)
    vs_ax.xaxis.set_label_coords(0.5, -0.3)
    vs_ax.yaxis.set_label_coords(-0.25, 0.5)
    sns.despine(ax=vs_ax)

    # Polar Hist =======================================================================================================
    # Convert radians to phase
    idx = vector_phase < 0
    vector_phase[idx] = vector_phase[idx] + 2 * np.pi
    idx = vector_phase_boot < 0
    vector_phase_boot[idx] = vector_phase_boot[idx] + 2 * np.pi
    # vs_hist_ax.grid(True, linewidth=0.2)
    # vs_hist_ax.axhline(linewidth=0.5, color="k")  # inc. width of y-axis and color it red
    vs_hist_ax.hist(vector_phase, bins=20, density=True, facecolor='k', alpha=1, label='Data')
    vs_hist_ax.hist(vector_phase_boot, bins=20, density=True, facecolor='0.5', alpha=0.75, label='Poisson')
    # xL = ['0', '', r'$\frac{\pi}{2}$', '', r'$\pi$', '', r'$\frac{3\pi}{2}$', '']
    vs_hist_ax.set_xlim(0, 2*np.pi+0.1)
    vs_hist_ax.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/2))

    xL = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'2$\pi$']
    vs_hist_ax.set_xticklabels(xL)
    vs_hist_ax.set_xlabel('Stimulus phase')

    # vs_hist_ax.legend(loc='upper right', shadow=False, frameon=False)
    # vs_hist_ax.arrow(0, 0, np.angle(mu), np.abs(mu), head_starts_at_zero=False, color='r', linewidth=2)
    # vs_hist_ax.plot(92, 1, 'r-')
    vs_hist_ax.set_ylim(0, 1)
    vs_hist_ax.xaxis.set_label_coords(0.5, -0.3)
    vs_hist_ax.set_yticks(np.arange(0, 1.1, 0.5))
    sns.despine(ax=vs_hist_ax)

    # Raster Plot ======================================================================================================
    # raster.set_title('Period = ' + info[2] + ' ms, Gap =' + info[0] + ' ms' + ', PD = ' + info[1] + ' ms, MFR: ' +
    #                  str(mean_rate) + ' Hz')
    for kk in range(len(spike_times)):
        # raster.plot(spike_times[kk], np.ones(len(spike_times[kk])) + kk, 'k|')
        for ii in range(len(spike_times[kk])-1):
            isi = spike_times[kk][ii+1] - spike_times[kk][ii]
            if isi < 0.012 and isi > 0.008 and mark_intervals:
                raster.plot([spike_times[kk][ii]*1000, spike_times[kk][ii+1]]*1000, [1 + kk, 1 + kk], 'g|--')
            else:
                raster.plot(spike_times[kk][ii]*1000, 1 + kk, 'k|')
    raster.set_xlim(0, x_limit*1000)
    raster.set_xticks(np.arange(0, x_limit * 1000+1, 50))
    raster.set_ylim(0, len(spike_times)+1)
    raster.set_yticks(np.arange(0, len(spike_times)+1, 5))
    raster.set_ylabel('Trial')
    raster.yaxis.set_label_coords(yaxis_pos1, 0.5)
    sns.despine(ax=raster)

    # PSTH Plot ========================================================================================================
    # # Binned
    # psth_ax.plot(bin_edges[:-1]*1000, frate, 'r', label='Firing Rate')
    # psth_ax.plot([0, bin_edges[-1]*1000], [overall_frate, overall_frate], 'b', label='Mean Firing Rate')
    # psth_ax.fill_between(bin_edges[:-1]*1000, frate - frate_std, frate + frate_std, facecolor='red', alpha=0.25)
    # psth_ax.fill_between([0, bin_edges[-1]*1000], overall_frate - overall_frate_std, overall_frate + overall_frate_std, facecolor='blue', alpha=0.25)

    # Convolved
    psth_ax.plot(rate_conv_time * 1000, rate_conv, 'k', label='Firing rate')
    psth_ax.plot([rate_conv_time[0] * 1000, rate_conv_time[-1] * 1000], [conv_overall_frate, conv_overall_frate], 'k--', label='Mean firing rate')
    psth_ax.fill_between(rate_conv_time * 1000, rate_conv - rate_conv_std, rate_conv + rate_conv_std, facecolor='k',
                         alpha=0.25)
    # psth_ax.fill_between([rate_conv_time[0] * 1000, rate_conv_time[-1] * 1000], conv_overall_frate - conv_overall_frate_std, conv_overall_frate + conv_overall_frate_std, facecolor='0.5',
    #                      alpha=0.25, hatch='//')
    psth_ax.plot([rate_conv_time[0] * 1000, rate_conv_time[-1] * 1000], [conv_overall_frate - conv_overall_frate_std, conv_overall_frate - conv_overall_frate_std], ':', color='k')
    psth_ax.plot([rate_conv_time[0] * 1000, rate_conv_time[-1] * 1000], [conv_overall_frate + conv_overall_frate_std, conv_overall_frate + conv_overall_frate_std], ':', color='k')

    psth_ax.set_xlim(0, x_limit*1000)
    psth_ax.set_xticks(np.arange(0, x_limit * 1000+1, 50))
    psth_ax.set_ylim(0, np.max(rate_conv+rate_conv_std))
    psth_ax.set_yticks(np.arange(0, np.max(rate_conv+rate_conv_std)+100, 100))
    psth_ax.set_ylabel('Firing rate [Hz]')
    psth_ax.legend(loc='upper right', shadow=False, frameon=False)
    psth_ax.yaxis.set_label_coords(yaxis_pos1, 0.5)
    sns.despine(ax=psth_ax)
    psth_ax.set_xlabel('Time [ms]')

    # Distances Profiles Plot ==========================================================================================
    # trains = [[]] * len(spike_times)
    # edges = [0, 1]
    # for q in range(len(spike_times)):
    #     trains[q] = spk.SpikeTrain(spike_times[q], edges)
    # isi_profile = spk.isi_profile(trains)
    # spike_profile = spk.spike_profile(trains)
    # sync_profile = spk.spike_sync_profile(trains)
    # isi_x, isi_y = isi_profile.get_plottable_data()
    # spike_x, spike_y = spike_profile.get_plottable_data()
    # sync_x, sync_y = sync_profile.get_plottable_data()
    #
    # # Running Average
    # N = 10
    # sync_ra = np.convolve(sync_y, np.ones((N,)) / N, mode='valid')
    # t_sync_ra = np.linspace(0, x_limit, len(sync_ra))
    # spike_ra = np.convolve(spike_y, np.ones((N,)) / N, mode='valid')
    # t_spike_ra = np.linspace(0, x_limit, len(spike_ra))
    # isi_ra = np.convolve(isi_y, np.ones((N,)) / N, mode='valid')
    # t_isi_ra = np.linspace(0, x_limit, len(isi_ra))
    #
    # distance.plot(stim_time*1000, stim, 'k-', label='Stimulus')
    # # distance.plot(t_sync_ra*1000, sync_ra, 'k', label='SYNC')
    # # distance.plot(t_spike_ra*1000, spike_ra, 'g', label='SPIKE')
    # # distance.plot(t_isi_ra*1000, isi_ra, 'm', label='ISI')
    #
    # # distance.plot(sync_x, sync_y, 'g', label='SYNC')
    # # distance.plot(isi_x, isi_y, 'k', label='ISI')
    # # distance.plot(spike_x, spike_y, 'r', label='SPIKE')
    # distance.set_xlim(0, x_limit*1000)
    # distance.set_xticks(np.arange(0, x_limit*1000+1, 50))
    # if np.min(stim) < -0.1:
    #     distance.set_ylim(-1, 1)
    #     distance.set_yticks(np.arange(-1, 1.1, 1))
    # else:
    #     distance.set_ylim(0, 1)
    #     distance.set_yticks(np.arange(0, 1.1, 1))
    # distance.set_ylabel('Stimulus')
    # # distance.legend(loc='upper right', shadow=False, frameon=True)
    # distance.yaxis.set_label_coords(yaxis_pos1, 0.5)
    # sns.despine(ax=distance)
    # distance.text(label_x_pos2, label_y_pos, 'a', transform=distance.transAxes, size=subfig_caps)
    # # distance.set_xlabel('Time [ms]')

    # Stimulus Plot ====================================================================================================
    stim1.plot(stim_time*1000, stim, 'k')
    stim1.set_xlim(0, x_limit*1000)
    stim1.set_xticks(np.arange(0, x_limit*1000+1, 50))
    stim1.set_axis_off()

    # Sub Caps
    raster.text(label_x_pos2, label_y_pos, 'a', transform=raster.transAxes, size=subfig_caps)
    psth_ax.text(label_x_pos2, label_y_pos, 'b', transform=psth_ax.transAxes, size=subfig_caps)
    isi_ax.text(label_x_pos1, label_y_pos, 'c', transform=isi_ax.transAxes, size=subfig_caps)
    vs_hist_ax.text(label_x_pos1+0.2, label_y_pos, 'd', transform=vs_hist_ax.transAxes, size=subfig_caps)
    vs_ax.text(label_x_pos1+0.1, label_y_pos, 'e', transform=vs_ax.transAxes, size=subfig_caps)

    # stim_trace.plot(stim_time*1000, stim, 'k')
    # stim_trace.set_xlim(0, x_limit)
    # stim_trace.set_yticks(np.arange(0, 1.1, 1))
    # stim_trace.set_xlabel('Time [s]')
    # stim_trace.set_ylabel('Stimulus')
    # stim_trace.yaxis.set_label_coords(-0.06, 0.5)
    # sns.despine(ax=stim_trace)
    # stim_trace.text(label_x_pos2, label_y_pos, 'a', transform=stim_trace.transAxes, size=subfig_caps)

    return 0


def spike_times_gap(path_names, protocol_name, show, save_data, th_factor=1, filter_on=True, window=None, mph_percent=2):
    """Spike Detection for Gap and Rect Paradigm

       Notes
       ----------
       This function detects spikes and plots the results with some additional statistics

       Parameters
       ----------
       path_names: Pathnames
       protocol_name:  protocol name (string)
       show: If true show plots (list: 0: spike detection, 1: overview, 2: vector strength)
       save_data: If true save spike times to HDD
       th_factor: Threshold for spike detection (th_factor * std)
       filter_on: If True data will be filtered with a band pass filter
       window: window size (seconds) for calculating threshold. If set to None the window has the length of the whole
       spike train
       mph_percent: Threshold for removing large spikes (B-cell). Threshold = mph_percent * std * mean

       Returns
       -------
       Spike Times (in seconds)

       """
    # Load Voltage Traces
    dataset = path_names[0]
    file_pathname = path_names[1]
    file_name = file_pathname + protocol_name + '_voltage.npy'
    voltage = np.load(file_name).item()
    if protocol_name is 'intervals_mas':
        tag_list = np.arange(0, len(voltage), 1)
    else:
        tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
        tag_list = np.load(tag_list_path)
    spikes = {}
    meta_data = {}
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Get Stimulus Information
    if protocol_name is 'intervals_mas':
        stim = [[]] * len(tag_list)
        stim_time = [[]] * len(tag_list)
    else:
        stim, stim_time, gap, pulse_duration, period = tagtostimulus_gap(path_names, protocol_name)

    # Loop trough all tags in tag_list
    for i in tqdm(range(len(tag_list)), desc='Spike Detection'):
        # if pulse_duration[i] == '10.0':
        #     continue
        try:
            if protocol_name is 'intervals_mas':
                trials = len(voltage[i]) - 5
                stim[i] = voltage[i]['stimulus']
                stim_time[i] = voltage[i]['stimulus_time']
            else:
                trials = len(voltage[tag_list[i]])
        except:
            continue
        spike_times = [list()] * trials
        spike_times_valley = [list()] * trials

        for k in range(trials):  # loop trough all trials
            if protocol_name is 'intervals_mas':
                try:
                    x = voltage[tag_list[i]][k][0]
                except KeyError:
                    print('Trial: ' + str(k) + ' not found')
                    continue
            else:
                x = voltage[tag_list[i]][k]
            if filter_on:
                nyqst = 0.5 * fs
                lowcut = 300
                highcut = 3000
                low = lowcut / nyqst
                high = highcut / nyqst
                x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)

            # Detect Spikes
            th = pk.std_threshold(x, fs, window, th_factor)
            spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

            # Remove large spikes
            t = np.arange(0, len(x) / fs, 1 / fs)
            spike_size = pk.peak_size_width(t, x, spike_times[k], spike_times_valley[k], pfac=0.75)
            spike_times[k], spike_times_valley[k], marked, marked_valley = \
                remove_large_spikes(x, spike_times[k], spike_times_valley[k], mph_percent=mph_percent, method='std')

            # Plot Spike Detection
            # Cut out spikes
            fs = 100 * 1000
            snippets = pk.snippets(x, spike_times[k], start=-100, stop=100)
            snippets_removed = pk.snippets(x, marked, start=-100, stop=100)

            if k == 0 and show is True:
                plot_spike_detection_gaps(x, spike_times[k], spike_times_valley[k], marked, spike_size, mph_percent, snippets,
                                          snippets_removed, th, window, tag_list[i], stim_time[i], stim[i])
                plt.show()

            spike_times[k] = spike_times[k] / fs  # in seconds
            spike_times_valley[k] = spike_times_valley[k] / fs  # in seconds
        spikes.update({tag_list[i]: spike_times})

        if protocol_name is 'intervals_mas':
            # Stim | Time | Gap | Tau | Freq
            metas = [voltage[i]['stimulus'], voltage[i]['stimulus_time'], voltage[i]['gap'], voltage[i][0][1], voltage[i][0][3]]
            meta_data.update({tag_list[i]: metas})

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_spikes.npy'
        np.save(file_name, spikes)
        if protocol_name is 'intervals_mas':
            file_name2 = file_pathname + protocol_name + '_meta.npy'
            np.save(file_name2, meta_data)
        print('Spike Times saved (protocol: ' + protocol_name + ')')

    return spikes


def spike_times_gap_spont(path_names, protocol_name, show, save_data, th_factor=1, filter_on=True, window=None, mph_percent=2):
    """Spike Detection for Gap and Rect Paradigm

       Notes
       ----------
       This function detects spikes and plots the results with some additional statistics

       Parameters
       ----------
       path_names: Pathnames
       protocol_name:  protocol name (string)
       show: If true show plots (list: 0: spike detection, 1: overview, 2: vector strength)
       save_data: If true save spike times to HDD
       th_factor: Threshold for spike detection (th_factor * std)
       filter_on: If True data will be filtered with a band pass filter
       window: window size (seconds) for calculating threshold. If set to None the window has the length of the whole
       spike train
       mph_percent: Threshold for removing large spikes (B-cell). Threshold = mph_percent * std * mean

       Returns
       -------
       Spike Times (in seconds)

       """
    # Load Voltage Traces
    dataset = path_names[0]
    file_pathname = path_names[1]
    file_name = file_pathname + protocol_name + '_spont_voltage.npy'
    voltage = np.load(file_name).item()
    if protocol_name is 'intervals_mas':
        tag_list = np.arange(0, len(voltage), 1)
    else:
        tag_list_path = file_pathname + protocol_name + '_spont_tag_list.npy'
        tag_list = np.load(tag_list_path)
    spikes = {}
    meta_data = {}
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Get Stimulus Information
    if protocol_name is 'intervals_mas':
        stim = [[]] * len(tag_list)
        stim_time = [[]] * len(tag_list)
    else:
        stim, stim_time, gap, pulse_duration, period = tagtostimulus_gap(path_names, protocol_name)

    # Loop trough all tags in tag_list
    for i in tqdm(range(len(tag_list)), desc='Spike Detection'):
        # if pulse_duration[i] == '10.0':
        #     continue
        try:
            if protocol_name is 'intervals_mas':
                trials = len(voltage[i]) - 5
                stim[i] = voltage[i]['stimulus']
                stim_time[i] = voltage[i]['stimulus_time']
            else:
                trials = len(voltage[tag_list[i]])
        except:
            continue
        spike_times = [list()] * trials
        spike_times_valley = [list()] * trials

        for k in range(trials):  # loop trough all trials
            if protocol_name is 'intervals_mas':
                try:
                    x = voltage[tag_list[i]][k][0]
                except KeyError:
                    print('Trial: ' + str(k) + ' not found')
                    continue
            else:
                x = voltage[tag_list[i]][k]
            if filter_on:
                nyqst = 0.5 * fs
                lowcut = 300
                highcut = 3000
                low = lowcut / nyqst
                high = highcut / nyqst
                x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)

            # Detect Spikes
            th = pk.std_threshold(x, fs, window, th_factor)
            spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

            # Remove large spikes
            t = np.arange(0, len(x) / fs, 1 / fs)
            spike_size = pk.peak_size_width(t, x, spike_times[k], spike_times_valley[k], pfac=0.75)
            spike_times[k], spike_times_valley[k], marked, marked_valley = \
                remove_large_spikes(x, spike_times[k], spike_times_valley[k], mph_percent=mph_percent, method='std')

            # Plot Spike Detection
            # Cut out spikes
            fs = 100 * 1000
            snippets = pk.snippets(x, spike_times[k], start=-100, stop=100)
            snippets_removed = pk.snippets(x, marked, start=-100, stop=100)

            if k == 0 and show is True:
                plot_spike_detection_gaps(x, spike_times[k], spike_times_valley[k], marked, spike_size, mph_percent, snippets,
                                          snippets_removed, th, window, tag_list[i], stim_time[i], stim[i])
                plt.show()

            spike_times[k] = spike_times[k] / fs  # in seconds
            spike_times_valley[k] = spike_times_valley[k] / fs  # in seconds
        spikes.update({tag_list[i]: spike_times})

        if protocol_name is 'intervals_mas':
            # Stim | Time | Gap | Tau | Freq
            metas = [voltage[i]['stimulus'], voltage[i]['stimulus_time'], voltage[i]['gap'], voltage[i][0][1], voltage[i][0][3]]
            meta_data.update({tag_list[i]: metas})

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_spont_spikes.npy'
        np.save(file_name, spikes)
        if protocol_name is 'intervals_mas':
            file_name2 = file_pathname + protocol_name + '_spont_meta.npy'
            np.save(file_name2, meta_data)
        print('Spike Times saved (protocol: ' + protocol_name + ')')

    return spikes


def remove_large_spikes(x, spike_times, spike_times_valley, mph_percent, method):
    # Remove large spikes
    if method == 'max':
        if spike_times.any():
            mask = np.ones(len(spike_times), dtype=bool)
            a = x[spike_times]
            idx_a = a >= np.max(a) * mph_percent
            mask[idx_a] = False
            marked = spike_times[idx_a]
            spike_times = spike_times[mask]

        if spike_times_valley.any():
            mask = np.ones(len(spike_times_valley), dtype=bool)
            b = x[spike_times_valley]
            idx_b = b <= np.min(b) * mph_percent
            mask[idx_b] = False
            marked_valley = spike_times_valley[idx_b]
            spike_times_valley = spike_times_valley[mask]
    if method == 'std':
        if spike_times.any():
            mask = np.ones(len(spike_times), dtype=bool)
            a = x[spike_times]
            a = a / np.max(a)
            idx_a = a >= np.mean(a) + np.std(a) * mph_percent
            mask[idx_a] = False
            marked = spike_times[idx_a]
            spike_times = spike_times[mask]
        else:
            marked = np.nan

        if spike_times_valley.any():
            mask = np.ones(len(spike_times_valley), dtype=bool)
            b = x[spike_times_valley]
            b = b / np.max(b)
            idx_b = b <= np.mean(b) + np.std(b) * mph_percent
            mask[idx_b] = False
            marked_valley = spike_times_valley[idx_b]
            spike_times_valley = spike_times_valley[mask]
        else:
            marked_valley = np.nan

    return spike_times, spike_times_valley, marked, marked_valley


def plot_detected_spikes(x, spike_times, spike_times_valley, marked, th, window, info, ax):
    """Plot Spike Detection
       Notes
       ----------
       This function plots spike detection.

       Parameters
       ----------
       x: Voltage Data
       spike_times: Spike Times
       spike_times_valley: Spike Times Valley
       marked: Spike Times that were removed (large spikes)
       th: Threshold for spike detection
       window: Window for calculating spiking threshold
       info: Information about stimulus
       ax: Axis for plotting
       Returns
       -------
       Plot
       """
    # plt.figure()
    fs = 100 * 1000
    t = np.arange(0, len(x) / fs, 1 / fs)
    ax.plot(t, x, 'k')
    # ax.set_xlabel('Time')
    ax.set_ylabel('Voltage [uV]')
    ax.plot(marked/fs, x[marked], 'kx')
    if spike_times.any():
        ax.plot(spike_times/fs, x[spike_times], 'ro')
    if spike_times_valley.any():
        ax.plot(spike_times_valley/fs, x[spike_times_valley], 'bo')
    if window is None:
        ax.plot([0, len(x)/fs], [th, th], 'r--')
        ax.plot([0, len(x)/fs], [-th, -th], 'r--')
        # ax.set_title(info + ', threshold=' + str(np.round(th, 2)))
    else:
        t_th = np.arange(0, len(th)/fs, 1/fs)
        ax.plot(t_th, th / 2, 'b--')
        ax.plot(t_th, -th / 2, 'b--')
        # ax.set_title(info + ', window=' + str(window))
    # plt.show()
    return 0


def spike_times_calls(path_names, protocol_name, show, save_data, th_factor=1, filter_on=True, window=0.1, mph_percent=0.8):
    """Get Spike Times using the thunderfish peak detection functions.

    Notes
    ----------
    This function gets all the spike times in the voltage traces loaded from HDD.

    Parameters
    ----------
    dataset :       Data set name (string)
    protocol_name:  protocol name (string)
    th_factor: threshold = th_factor * median(abs(x)/0.6745)
    filter_on: True: Filter Data with bandpass filter
    window: Window size for thresholding
    mph_percent: Remove all spikes with amplitude higher than max. spike amplitude * mph_percent
    show: Show spike detection graphically
    save_data: Save data to HDD
    Returns
    -------
    spikes: Saves spike times (in seconds) to HDD in a .npy file (dict).

    """

    # Load Voltage Traces
    save_fig = False
    dataset = path_names[0]
    file_pathname = path_names[1]
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    tag_list = np.load(tag_list_path)
    spikes = {}
    fs = 100*1000  # Sampling Rate of Ephys Recording

    _, connections = tagtostimulus(path_names)

    # Loop trough all tags in tag_list
    for i in tqdm(range(len(tag_list)), desc='Spike Detection'):
        trials = len(voltage[tag_list[i]])
        spike_times = [list()] * trials
        spike_times_valley = [list()] * trials
        for k in range(trials):  # loop trough all trials
            # Filter Voltage Trace
            x = voltage[tag_list[i]][k]
            if filter_on:
                nyqst = 0.5 * fs
                lowcut = 300
                highcut = 2000
                low = lowcut / nyqst
                high = highcut / nyqst
                x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)

            th = pk.std_threshold(x, fs, window, th_factor)
            spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

            # Remove large spikes
            t = np.arange(0, len(x) / fs, 1 / fs)
            spike_size = pk.peak_size_width(t, x, spike_times[k], spike_times_valley[k], pfac=0.75)
            # [idx][time, height, size, width, count]

            spike_times[k], spike_times_valley[k], marked, marked_valley = \
                remove_large_spikes(x, spike_times[k], spike_times_valley[k], mph_percent=2, method='std')

            # Plot detected spikes of random trials
            # rand_plot = np.random.randint(100)
            # if rand_plot >= 99 and show:
            if k == 0 and (i == 0 or i == 80) and show:  #and connections[tag_list[i]].startswith('calls'):
                # Cut out spikes
                snippets = pk.snippets(x, spike_times[k], start=-100, stop=100)
                snippets_removed = pk.snippets(x, marked, start=-100, stop=100)

                fig = plt.figure(figsize=(12, 8))
                fig_size = (3, 2)
                spike_shapes = plt.subplot2grid(fig_size, (0, 0), rowspan=1, colspan=1)
                spike_width = plt.subplot2grid(fig_size, (0, 1), rowspan=1, colspan=1)
                volt_trace = plt.subplot2grid(fig_size, (1, 0), rowspan=1, colspan=2)
                stim_trace = plt.subplot2grid(fig_size, (2, 0), rowspan=1, colspan=2)

                t_snippets = np.arange(-100/fs, 100/fs, 1/fs)*1000
                for s in range(len(snippets)):
                    spike_shapes.plot(t_snippets, snippets[s], 'k')
                for s in range(len(snippets_removed)):
                    spike_shapes.plot(t_snippets, snippets_removed[s], 'r')

                spike_shapes.set_xlabel('Time [ms]')
                spike_shapes.set_ylabel('Voltage [uV]')

                # Width vs Size
                spike_width.plot(spike_size[:, 3]*1000, spike_size[:, 2], 'ko')
                spike_width.set_xlabel('Spike Width [ms]')
                spike_width.set_ylabel('Spike Size [uV]')

                # Width vs Height
                ax2 = spike_width.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.plot(spike_size[:, 3] * 1000, spike_size[:, 1], 'gx')
                ax2.plot([np.min(spike_size[:, 3] * 1000), np.max(spike_size[:, 3] * 1000)],
                         [np.max(spike_size[:, 1]) * mph_percent, np.max(spike_size[:, 1]) * mph_percent], 'g--')
                ax2.set_ylabel('Spike Height[uV]', color='g')
                ax2.tick_params(axis='y', labelcolor='g')

                sound_file = wav.read('/media/brehm/Data/MasterMoth/stimuli_backup/' + connections[tag_list[i]])
                t_sound = np.arange(0, len(sound_file[1])/sound_file[0], 1/sound_file[0])

                plot_detected_spikes(x, spike_times[k], spike_times_valley[k], marked, th, window, connections[tag_list[i]], volt_trace)
                volt_trace.set_xticks([])

                stim_trace.plot(t_sound*1000, sound_file[1], 'k')
                stim_trace.set_xlabel('Time [ms]')
                stim_trace.set_yticks([])
                fig.tight_layout()
                plt.show()

            spike_times[k] = spike_times[k] / fs  # in seconds
        spikes.update({tag_list[i]: spike_times})
        if False:
            if connections[tag_list[i]].startswith('nat'):
                x_limit = 0.4
            if connections[tag_list[i]].startswith('batcall'):
                x_limit = 0.4
            if connections[tag_list[i]].startswith('calls'):
                x_limit = 3

            sound_file = wav.read('/media/brehm/Data/MasterMoth/stimuli/' + connections[tag_list[i]])
            t_sound = np.arange(0, len(sound_file[1]) / sound_file[0], 1 / sound_file[0])
            bin_size = 0.002
            f_rate, bin_edges = psth(spike_times, bin_size, plot=False, return_values=True, tmax=1, tmin=0)
            # f_rate, bin_edges = psth(spike_times, len(spike_times), bin_size, plot=False, return_values=True, separate_trials=True)
            plt.figure()
            tr = 3
            for w in range(tr):
                plt.subplot(tr+3, 1, w+1)
                if w == 0:
                    plt.title(connections[tag_list[i]])
                x = voltage[tag_list[i]][w+1]
                t = np.arange(0, len(x) / fs, 1 / fs)
                plt.plot(t, x)
                idx = spike_times[w+1] * fs
                plt.plot(spike_times[w+1], x[idx.astype(int)], 'ro')
                plt.ylabel('Volt [uV]')
                plt.xticks([])
                plt.xlim(0, x_limit)

            plt.subplot(tr+3, 1, tr+1)
            plt.plot(bin_edges[:-1], f_rate, 'k')
            plt.ylabel('Firing Rate [Hz]')
            plt.xticks([])
            plt.xlim(0, x_limit)

            plt.subplot(tr+3, 1, tr+2)
            for kk in range(len(spike_times)):
                plt.plot(spike_times[kk], np.ones(len(spike_times[kk])) + kk, 'k|', 'LineWidth', 4)
            plt.xticks([])
            plt.ylabel('Trials')
            plt.xlim(0, x_limit)

            plt.subplot(tr+3, 1, tr+3)
            plt.plot(t_sound, sound_file[1], 'k')
            plt.xlabel('Time [s]')
            plt.yticks([])
            plt.xticks(np.arange(0, x_limit, 0.1))
            plt.xlim(0, x_limit)

            if save_fig:
                # Save Plot to HDD
                sp = path_names[2] + 'SpikeDetection/'
                fig = plt.gcf()
                fig.set_size_inches(15, 10)
                fig.savefig(sp + connections[tag_list[i]] + '_SpikeDetection.png', bbox_inches='tight', dpi=150)
                plt.close(fig)
            else:
                plt.show()

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_spikes.npy'
        np.save(file_name, spikes)
        print('Spike Times saved (protocol: ' + protocol_name + ')')

    return spikes

# ----------------------------------------------------------------------------------------------------------------------
# FILTER


def voltage_trace_filter(voltage, cutoff, order, ftype, filter_on=True):
    if filter_on:
        b, a = sg.butter(order, cutoff, btype=ftype, analog=False)
        y = sg.filtfilt(b, a, voltage)
    else:
        y = voltage
    return y


# ----------------------------------------------------------------------------------------------------------------------
# Reconstruct Stimuli


def rect_stimulus(period, pulse_duration, stimulus_duration, total_amplitude, sampling_rate, plotting):
    # Compute square wave stimulus in time.
    # Input needs to be in seconds.
    # plotting = True will plot the stimulus
    # Returns:
    #           - Array containing the amplitude
    #           - Array containing time

    """
    # This is in samples (low resolution)
    sampling_rate = 800*1000
    stimulus_duration = stimulus_duration*1000
    period = period*1000  # in ms
    pulse_duration = pulse_duration*1000

    stimulus = np.zeros((int(stimulus_duration), 1))
    pulse_times = np.arange(0,500,int(period))

    for i in pulse_times:
        stimulus[i:i+int(pulse_duration)] = 1

    plt.plot(stimulus)
    plt.show()
    """
    # This is with sampling rate (high resolution) and in time
    stimulus = np.zeros((int(stimulus_duration*sampling_rate), 1))
    stimulus_time = np.linspace(0,stimulus_duration, sampling_rate*stimulus_duration)
    pulse_times = np.arange(0, stimulus_duration*sampling_rate, period*sampling_rate)
    for i in pulse_times:
        stimulus[int(i):int(i)+int(pulse_duration*sampling_rate)] = 1

    stimulus = stimulus*total_amplitude
    if plotting:
        plt.plot(stimulus_time, stimulus)
        plt.show()

    return stimulus_time, stimulus


# ----------------------------------------------------------------------------------------------------------------------
# NIX Functions


def get_voltage_trace(path_names, tag, protocol_name, multi_tag, search_for_tags):
    """Get Voltage Trace from nix file.

    Notes
    ----------
    This function reads out the voltage traces for the given tag names stored in the nix file

    Parameters
    ----------
    dataset :       Data set name (string)
    tag:            Tag name (string)
    protocol_name:  protocol name (string) (only important for saving the data)
    search_for_tags: search for all tags containing the string in tag. If set to false, the list in 'tag' is used.
    multi_tag: If true then function treats tags as multi tags and looks for all trials.

    Returns
    -------
    voltage: Saves Voltage Traces to HDD in a .npy file (dict)

    """

    dataset = path_names[0]
    file_pathname = path_names[1]
    nix_file = path_names[3] + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    if search_for_tags:
        if multi_tag:
            tag_list = [t.name for t in b.multi_tags if tag in t.name]  # Find Multi-Tags
        else:
            tag_list = [t.name for t in b.tags if tag in t.name]  # Find Tags
    else:
        tag_list = tag

    sampling_rate = 100*1000
    voltage = {}
    if multi_tag:
        for i in range(len(tag_list)):
            # Get tags
            mtag = b.multi_tags[tag_list[i]]

            trials = len(mtag.positions[:])  # Get number of trials
            volt = np.zeros((trials, int(np.ceil(mtag.extents[0]*sampling_rate))))  # allocate memory
            for k in range(trials):  # Loop through all trials
                v = mtag.retrieve_data(k, 0)[:]  # Get Voltage for each trial
                volt[k, :len(v)] = v
            voltage.update({tag_list[i]: volt})  # Store Stimulus Name and voltage traces in dict
    else:
        for i in range(len(tag_list)):
            # Get tags
            mtag = b.tags[tag_list[i]]
            volt = mtag.retrieve_data(0)[:]  # Get Voltage
            voltage.update({tag_list[i]: volt})  # Store Stimulus Name and voltage traces in dict


    # Save to HDD
    file_name = file_pathname + protocol_name + '_voltage.npy'
    np.save(file_name, voltage)
    print('Voltage Traces saved (protocol: ' + protocol_name + ')')

    file_name2 = file_pathname + protocol_name + '_tag_list.npy'
    np.save(file_name2, tag_list)
    print('Tag List saved (protocol: ' + protocol_name + ')')
    f.close()
    return voltage, tag_list


def get_voltage_trace_gap(path_names, tag, protocol_name, multi_tag, search_for_tags, save_data):
    """Get Voltage Trace from nix file.

    Notes
    ----------
    This function reads out the voltage traces for the given tag names stored in the nix file

    Parameters
    ----------
    dataset :       Data set name (string)
    tag:            Tag name (string)
    protocol_name:  protocol name (string) (only important for saving the data)
    search_for_tags: search for all tags containing the string in tag. If set to false, the list in 'tag' is used.
    multi_tag: If true then function treats tags as multi tags and looks for all trials.

    Returns
    -------
    voltage: Saves Voltage Traces to HDD in a .npy file (dict)

    """

    dataset = path_names[0]
    file_pathname = path_names[1]
    nix_file = path_names[3] + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    if search_for_tags:
        if multi_tag:
            tag_list = [t.name for t in b.multi_tags if tag in t.name]  # Find Multi-Tags
        else:
            tag_list = [t.name for t in b.tags if tag in t.name]  # Find Tags
    else:
        tag_list = tag

    # Find all multi tags that are linked to the tags (input):
    m = 'SingleStimulus-square_wave-sine_wave-'
    mtag_list = [t.name for t in b.multi_tags if m in t.name]  # Find Multi-Tags

    sampling_rate = 100*1000

    tag1 = b.tags[tag_list[0]].position[0]
    tag2 = b.tags[tag_list[-1]].position[0] + b.tags[tag_list[-1]].extent[0]
    mtags = []

    for k in range(len(mtag_list)):
        try:
            pos = b.multi_tags[mtag_list[k]].positions[0]
            if pos > tag1 and pos < tag2:
                mtags.append(mtag_list[k])
        except:
            print(mtag_list[k] + ' not found')

    if len(tag_list) > len(mtags):
        print('Did you run this protocol more than once?')

    voltage = {}
    for i in tqdm(range(len(mtags)), desc='Get Voltage'):
        # Get tags
        mtag = b.multi_tags[mtags[i]]
        trials = len(mtag.positions[:])  # Get number of trials
        volt = np.zeros((trials, int(np.ceil(mtag.extents[0]*sampling_rate))))  # allocate memory
        for k in range(trials):  # Loop through all trials
            v = mtag.retrieve_data(k, 0)[:]  # Get Voltage for each trial
            volt[k, :len(v)] = v
        voltage.update({tag_list[i]: volt})  # Store Stimulus Name and voltage traces in dict
        # voltage.update({mtags[i]: volt})  # Store Stimulus Name and voltage traces in dict

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_voltage.npy'
        np.save(file_name, voltage)
        print('Voltage Traces saved (protocol: ' + protocol_name + ')')

        file_name2 = file_pathname + protocol_name + '_tag_list.npy'
        np.save(file_name2, tag_list)
        print('Tag List saved (protocol: ' + protocol_name + ')')
    f.close()
    return voltage, tag_list


def get_voltage_trace_gap_spontan(path_names, tag, protocol_name, multi_tag, search_for_tags, save_data):
    """Get Voltage Trace from nix file.

    Notes
    ----------
    This function reads out the voltage traces for the given tag names stored in the nix file

    Parameters
    ----------
    dataset :       Data set name (string)
    tag:            Tag name (string)
    protocol_name:  protocol name (string) (only important for saving the data)
    search_for_tags: search for all tags containing the string in tag. If set to false, the list in 'tag' is used.
    multi_tag: If true then function treats tags as multi tags and looks for all trials.

    Returns
    -------
    voltage: Saves Voltage Traces to HDD in a .npy file (dict)

    """

    dataset = path_names[0]
    file_pathname = path_names[1]
    nix_file = path_names[3] + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    if search_for_tags:
        if multi_tag:
            tag_list = [t.name for t in b.multi_tags if tag in t.name]  # Find Multi-Tags
        else:
            tag_list = [t.name for t in b.tags if tag in t.name]  # Find Tags
    else:
        tag_list = tag

    # Find all multi tags that are linked to the tags (input):
    m = 'SingleStimulus-square_wave-sine_wave-'
    mtag_list = [t.name for t in b.multi_tags if m in t.name]  # Find Multi-Tags

    sampling_rate = 100*1000

    tag1 = b.tags[tag_list[0]].position[0]
    tag2 = b.tags[tag_list[-1]].position[0] + b.tags[tag_list[-1]].extent[0]
    mtags = []

    for k in range(len(mtag_list)):
        try:
            pos = b.multi_tags[mtag_list[k]].positions[0]
            if pos > tag1 and pos < tag2:
                mtags.append(mtag_list[k])
        except:
            print(mtag_list[k] + ' not found')

    if len(tag_list) > len(mtags):
        print('Did you run this protocol more than once?')

    voltage = {}
    for i in tqdm(range(len(mtags)), desc='Get Voltage'):
        # Get tags
        mtag = b.multi_tags[mtags[i]]
        look_back = 0.1
        look_future = 0.1
        dat = b.data_arrays[mtag.references[0].name]
        dim = dat.dimensions[0]
        trials = len(mtag.positions[:])  # Get number of trials
        volt = np.zeros((trials, int(np.ceil((mtag.extents[0]+look_future+look_back)*sampling_rate))))  # allocate memory
        for k in range(trials):  # Loop through all trials
            idx = int(dim.index_of(mtag.positions[k] - look_back))
            idx2 = int(dim.index_of(mtag.positions[k] + mtag.extents[k] + look_future))
            # idx_extent = int(dim.index_of(mtag.positions[k]))
            v = dat[idx:idx2]
            # v = mtag.retrieve_data(k, 0)[:]  # Get Voltage for each trial
            volt[k, :len(v)] = v
        voltage.update({tag_list[i]: volt})  # Store Stimulus Name and voltage traces in dict
        # voltage.update({mtags[i]: volt})  # Store Stimulus Name and voltage traces in dict

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_spont_voltage.npy'
        np.save(file_name, voltage)
        print('Voltage Traces saved (protocol: ' + protocol_name + ')')

        file_name2 = file_pathname + protocol_name + '_spont_tag_list.npy'
        np.save(file_name2, tag_list)
        print('Tag List saved (protocol: ' + protocol_name + ')')
    f.close()
    return voltage, tag_list


def get_metadata(dataset, tag, protocol_name):
    """Get MetaData.

    Notes
    ----------
    This function reads out the metadata for a given protocol (list of tag names)
    Metadata is then saved as a .npy file in the nix data structure:
    mtags.metadata.sections[0][x]

    Parameters
    ----------
    dataset :       Data set name (string)
    tag:            Tag name (string)
    protocol_name:  protocol name (string)

    Returns
    -------
    mtags: nix mtag is saved which includes all the meta data in a .npy file (nix format)

    """

    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + dataset + '/' + dataset + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    tag_list = [t.name for t in b.multi_tags if tag in t.name]  # Find Tags
    mtags = {}
    for k in range(len(tag_list)):
        mtags.update({tag_list[k]: b.multi_tags[tag_list[k]]})

    # This function does not close the nix file! Make sure that you close the nix file after using this function
    # Save to HDD
    #file_name = file_pathname + protocol_name + '_metadata.npy'
    #np.save(file_name, mtags)
    #print('Metadata saved (protocol: ' + protocol_name + ')')

    return f, mtags


def list_protocols(path_names, protocol_name, tag_name, save_txt):
    """List all protocols in nix file

    Notes
    ----------
    This function looks for specific tags and multi tags in a nix file and lists all the protocols that were found. It
    also gives the tag names that include the protcol that you are looking for.

    Parameters
    ----------
    dataset :       Data set name (string)
    protocol_name:  protocol name (string)
    tag_name: Name of tag that you are looking for (string)
    save_txt: If true all gathered information will be saved to a text file

    Returns
    -------
    target: List of tag names that have the protocol you're looking for
    p: List of protocols that were found in tags
    mtarget: List of multi tag names that have the protocol you're looking for
    mp: List of protocols that were found in multi tags

    """
    data_name = path_names[0]
    file_pathname = path_names[1]
    nix_file = path_names[3] + '.nix'

    # Try to open the nix file
    try:
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        print('".nix" extension found')
        print(data_name)
    except RuntimeError:
        try:
            f = nix.File.open(nix_file + '.h5', nix.FileMode.ReadOnly)
            print('".nix.h5" extension found')
            print(data_name)
        except RuntimeError:
            print(data_name)
            print('File not found')
            return 1
    b = f.blocks[0]

    # Search for tags and multi tags
    tag_list = [t.name for t in b.tags if tag_name[0] in t.name]
    mtag_list = [t.name for t in b.multi_tags if tag_name[1] in t.name]
    target = []
    mtarget = []

    # Create Text File:
    if save_txt:
        text_file_name = file_pathname + 'stimulus_protocols.txt'
        text_file_name2 = file_pathname + 'stimulus_songs.txt'
        try:
            text_file = open(text_file_name, 'r+')
            text_file2 = open(text_file_name2, 'r+')
            text_file.truncate(0)
            text_file2.truncate(0)
        except FileNotFoundError:
            print('create new txt files')

    # Get the Metadata for the Tags:
    p = [''] * len(tag_list)
    try:
        for i in range(len(tag_list)):
            p[i] = b.tags[tag_list[i]].metadata.sections[0].sections[0].props[1].name
            if p[i] == protocol_name:
                target.append(tag_list[i])

        # Write to txt file
        if save_txt:
            with open(text_file_name, 'a') as text_file:
                text_file.write(data_name + '\n')
                text_file.write(tag_name[0] + '\n\n')
                for k in range(len(p)):
                    text_file.write(p[k] + '\n')
    except KeyError:
        print('No Protocols found')

    # Get the Metadata for the Multi Tags:
    mp = [''] * len(mtag_list)
    try:
        for i in range(len(mtag_list)):
            mp[i] = b.multi_tags[mtag_list[i]].metadata.sections[0][2]  # This is the sound file name
            if mp[i] == protocol_name:
                mtarget.append(mtag_list[i])
        # Write to txt file
        if save_txt:
            with open(text_file_name2, 'a') as text_file:
                text_file.write(data_name + '\n')
                text_file.write(tag_name[1] + '\n\n')
                for k in range(len(mp)):
                    text_file.write(mp[k] + '\n')
    except KeyError:
        print('No Songs found')

    # Close the nix file
    f.close()
    return target, p, mtarget, mp


def overview_recordings(look_for_mtags):

    # List all recordings
    recs = os.listdir('/media/brehm/Data/MasterMoth/mothdata')
    recs = [s for s in recs if '2' in s]
    not_found = []
    # Create csv
    with open('overview.csv', 'w', newline='') as csvfile:
        textwriter = csv.writer(csvfile, delimiter=',')
        textwriter.writerow(['Recording', 'Gap', 'PulseIntervalRect', 'MAS', 'FI', 'Calls', 'Species', 'Sex', 'Age'])

    for data_name in tqdm(recs, desc='Data Files'):
        file_pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/DataFiles/"
        nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + data_name + '.nix'

        # Open the nix file
        try:
            f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
            # print('".nix" extension found')
            # print(data_name)
        except:
            try:
                f = nix.File.open(nix_file + '.h5', nix.FileMode.ReadOnly)
                # print('".nix.h5" extension found')
                # print(data_name)
            except:
                # print(data_name)
                # print('File not found')
                not_found.append(data_name)
                continue
        b = f.blocks[0]

        # Get all tags:
        tags = [[]] * len(b.tags)
        tag_protocols = [[]] * len(b.tags)
        k = 0
        for i in b.tags:
            tags[k] = i.name
            try:
                # Get Macro Name
                tag_protocols[k] = b.tags[i.name].metadata.sections[0].sections[0].props[1].name
            except:
                tag_protocols[k] = 'None'
            k += 1

        # Get all multi tags:
        if look_for_mtags:
            multi_tags = [[]] * len(b.multi_tags)
            mtag_protocols = [[]] * len(b.multi_tags)
            mtag_metadata = {}
            k = 0
            for i in tqdm(b.multi_tags, desc='Multi Tags'):
                multi_tags[k] = i.name
                try:
                    # Get all metadata
                    mtag_protocols[k] = getMetadataDict(b.multi_tags[i.name])
                    mtag_metadata.update({i.name: getMetadataDict(b.multi_tags[i.name])})
                except:
                    mtag_protocols[k] = 'None'
                k += 1

        # Check what the recording has to offer
        # macros: Gap | PulseIntervalRect | MAS | FI | Calls
        macros = [False] * 5
        if any("Gap" in s for s in tag_protocols):
            macros[0] = True
        if any("PulseIntervalsRect" in s for s in tag_protocols):
            macros[1] = True
        # if any(len("PulseIntervals") == len(s) for s in tag_protocols) and any("PulseIntervals" in s for s in tag_protocols):
        #     macros[2] = True
        # if any("FIField" in s for s in tag_protocols):
        #     macros[3] = True
        if any("Moth" in s for s in tag_protocols):
            macros[4] = True

        if any("MothA" in s for s in tags):
            macros[2] = True
        if any("FI" in s for s in tags):
            macros[3] = True

        # Get Animal Species
        try:
            meta_rec = getMetadataDict(b)
            animal = meta_rec['Subject']['Species'][0]
            age = meta_rec['Subject']['Age'][0]
            sex = meta_rec['Subject']['Sex'][0]
        except:
            animal = 'Unknown'
            age = 'Unknown'
            sex = 'Unknown'
        # Write Info to csv
        with open('overview.csv', 'a', newline='') as csvfile:
            textwriter = csv.writer(csvfile, delimiter=',')
            # textwriter.writerow(['Recording', 'Gap', 'PulseIntervalRect', 'MAS', 'FI', 'Calls', 'Species', 'Sex', 'Age'])
            textwriter.writerow([data_name] + macros + [animal, sex, age])
        # Close nix file
        f.close()

    print('This files could not be found or opend:')
    for i in not_found: print(i)
    return 0

# ----------------------------------------------------------------------------------------------------------------------
# SPIKE STATISTICS


def convolution_rate(spikes, t_max, dt, sigma, method='mean'):
    def gauss_kernel(s, step):
        x = np.arange(-4 * s, 4 * s, step)
        y = np.exp(-0.5 * (x/s)**2) / (np.sqrt(2 * np.pi) * s)
        # y = np.exp(-(x**2 / 2*s**2)) / (np.sqrt(2 * np.pi) * s)
        return y

    rate = [[]] * len(spikes)
    for k in range(len(spikes)):
        spike_times = spikes[k][spikes[k] < t_max]
        # t = np.arange(0, t_max-dt, dt)
        t = np.arange(0, t_max, dt)
        r = np.zeros(len(t))
        spike_ids = np.round(spike_times / dt) - 1
        try:
            r[spike_ids.astype('int')] = 1
        except:
            print('Error in convolution rate')
            embed()
            exit()
        kernel = gauss_kernel(sigma, dt)
        rate[k] = np.convolve(r, kernel, mode='same')

    if method is 'max':
        max_rate = np.max(rate, axis=1)
    if method is 'mean':
        max_rate = np.mean(rate, axis=1)

    firing_rate = np.mean(rate, axis=0)
    std = np.std(rate, axis=0)
    mean_max_rate = np.mean(max_rate)
    std_max_rate = np.std(max_rate)

    return t, firing_rate, std, mean_max_rate, std_max_rate


def inter_spike_interval(spikes):
    isi = np.diff(spikes)
    return isi


def psth(spike_times, bin_size, plot, return_values, tmax, tmin):
    # spike times must be seconds!
    # Compute histogram and calculate time dependent firing rate (binned PSTH)
    # n: number of trials
    # bin_size: bin size in seconds
    bins = int(tmax / bin_size)
    hist = [[]] * len(spike_times)
    for i in range(len(spike_times)):
        sp = spike_times[i][spike_times[i] > tmin]
        hist[i], bin_edges = np.histogram(sp, bins)
    hist = np.array(hist)
    mean_f_rate = np.mean(hist, axis=0) / bin_size

    if plot:
        # Plot PSTH
        plt.plot(bin_edges[:-1], mean_f_rate, 'k')

    if return_values:
        return mean_f_rate, bin_edges
    else:
        return 0


def raster_plot(sp_times, stimulus_time, steps):
    for k in range(len(sp_times)):
        plt.plot(sp_times[k], np.ones(len(sp_times[k])) + k, 'k|')

    # plt.xlim(0, stimulus_time[-1])
    plt.xticks(np.arange(0, stimulus_time[-1], steps))
    plt.yticks(np.arange(0, len(sp_times) + 1, 5))
    plt.ylabel('Trial Number')

    return 0


def plot_isi_histogram(spike_times, p_spike_times, bin_size, x_limit, steps, info, intervals, tmin, plot_title, ax):
    # Compute Inter Spike Intervals
    isi = [[]] * len(spike_times)
    for i in range(len(spike_times)):
        sp = spike_times[i][spike_times[i] > tmin]
        isi[i] = np.diff(sp)

    if len(intervals) > 1:
        isi_p = intervals
    else:
        isi_p = [[]] * len(p_spike_times)
        for k in range(len(p_spike_times)):
            sp_p = p_spike_times[i][p_spike_times[i] > tmin]
            isi_p[k] = np.diff(sp_p)

    # Convert to ms
    isi = np.concatenate(isi)
    isi = isi * 1000
    isi_p = np.concatenate(isi_p)
    isi_p = isi_p * 1000

    # Cut off the first part

    # Compute Histograms
    bins_nr1 = int(np.max(isi) / bin_size)
    bins_nr2 = int(np.max(isi_p) / bin_size)
    n1, bins1, patches1 = ax.hist(isi, bins_nr1, density=True, facecolor='k', alpha=1, label='Data')
    n2, bins2, patches2 = ax.hist(isi_p, bins_nr2, density=True, facecolor='0.5', alpha=0.75, label='Poisson')

    ax.set_xlabel('ISI [ms]')
    ax.set_ylabel('Prob. density')
    ax.set_ylim(0, np.max(n1)+0.05)
    ax.set_yticks(np.arange(0, np.max(n1)+0.1, 0.1))
    # ax.set_xlim(0, x_limit)
    # ax.set_xticks(np.arange(0, x_limit, steps))
    if plot_title:
        plt.title('period: ' + info[0] + ' ms, gap: ' + info[1] + 'ms')
    #ax.set_legend(['Data', 'Poisson'])

    return 0


def vs_range(spike_times, pp, tmin, n_ci, order):

    if order == 2:
        vs = np.zeros(shape=(len(spike_times), len(pp)))
        phase = np.zeros(shape=(len(spike_times), len(pp)))
        # Compute Vector Strength for all periods in pp
        for q in range(len(spike_times)):
            vs[q, :], phase[q, :] = sg.vectorstrength(spike_times[q][spike_times[q] > tmin], pp)
        # Compute desciptive statistics
        vs_mean = np.nanmean(vs, axis=0)
        vs_percentile = np.nanpercentile(vs, 95, axis=0)
        vs_std = np.nanstd(vs, axis=0)

        # Compute CI for every period (gap)
        vs_ci = np.zeros(shape=(len(pp), 2))
        if n_ci > 0:
            for i in range(len(pp)):
                vs_ci[i, :] = bootstrap_ci(vs[:, i], n_boot=n_ci, level=95)
        else:
            vs_ci = np.nan

    if order == 1:
        sp = np.concatenate(spike_times)
        vs_mean, phase = sg.vectorstrength(sp[sp > tmin], pp)
        vs_std = np.nan
        vs_percentile = np.nan
        vs_ci = np.nan
        vs = np.nan

    # Find Peaks in VS Distribution
    # th = pk.std_threshold(vs_mean, th_factor=2)
    # peaks, _ = pk.detect_peaks(vs_mean, th)
    return vs, phase, vs_mean, vs_std, vs_percentile, vs_ci


def dPrime(hits, misses, fas, crs):
    Z = norm.ppf
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5 / (hits + misses)
    halfFa = 0.5 / (fas + crs)

    # Calculate hitrate and avoid d' infinity
    hitRate = hits / (hits + misses)
    if hitRate == 1:
        hitRate = 1 - halfHit
    if hitRate == 0:
        hitRate = halfHit

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas / (fas + crs)
    if faRate == 1:
        faRate = 1 - halfFa
    if faRate == 0:
        faRate = halfFa

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp((Z(faRate) ** 2 - Z(hitRate) ** 2) / 2)
    out['c'] = -(Z(hitRate) + Z(faRate)) / 2
    out['Ad'] = norm.cdf(out['d'] / sqrt(2))
    out['hit_rate'] = hitRate
    out['fa_rate'] = faRate
    return out

# ----------------------------------------------------------------------------------------------------------------------
# FI FIELD:


def plot_ficurve(amplitude_sorted, mean_spike_count, std_spike_count, freq, spike_threshold, pathname, savefig):
    plt.figure()
    plt.errorbar(amplitude_sorted, mean_spike_count, yerr=std_spike_count, fmt='k-o')
    plt.plot(amplitude_sorted, np.zeros((len(amplitude_sorted)))+spike_threshold, 'r--')
    plt.xlabel('Stimuli Amplitude [dB SPL]')
    plt.ylabel('Spikes per Stimuli')
    plt.xlim(10, np.max(amplitude_sorted)+10)
    plt.title(('At %s kHz' % freq))
    if not savefig:
        plt.show()

    # Save Plot to HDD
    if savefig:
        figname = pathname + "FICurve_" + str(freq) + "kHz.png"
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return 'FICurve saved'


def plot_fifield(db_threshold, pathname, savefig):
    # Plot FI Field
    plt.figure()
    plt.plot(db_threshold[:, 0], db_threshold[:, 1], 'k-o')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Threshold [dB SPL]')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 100, 10))
    if not savefig:
        plt.show()

    # Save Plot to HDD
    if savefig:
        figname = pathname + "FIField.png"
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return 'FIField saved'


def fifield_voltage2(path_name, rec_dur_factor):
    data_name = path_name[0]
    nix_file = path_name[3] + '.nix'
    data_files = path_name[1]

    try:
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        b = f.blocks[0]
    except:
        print('Could not open nix file')
        return 0

    tag_list = []
    fi_tag = []
    for t in b.multi_tags:
        tag_list.append(t.name)
        if t.name.startswith('FIField'):
            fi_tag.append(t.name)
    try:
        # Get tags
        mtag = b.multi_tags[fi_tag[0]]
    except:
        print('No FIField found')
        print(fi_tag)
        return 0

    # a = input('Conitnue?')
    # if a is '0':
    #     exit()
    # if a is '1':
    #     embed()
    #     return 0

    # Get Meta Data
    # meta = mtag.metadata.sections[0]
    # duration = meta["Duration"]
    frequency = mtag.features[5].data[:]
    amplitude = mtag.features[4].data[:]

    volt = [[]] * len(frequency)
    parameters = np.zeros(shape=(len(frequency), 3))

    # Get Data Array
    dat = b.data_arrays[ mtag.references[0].name]
    dim = dat.dimensions[0]
    tag_end = mtag.extents[0]
    dur_of_collecting = rec_dur_factor * tag_end
    for k in tqdm(range(len(frequency)), desc='Collecting Voltage Traces'):
        idx = int(dim.index_of(mtag.positions[k]))
        idx_extent = int(dim.index_of(mtag.positions[k] + dur_of_collecting))
        volt[k] = dat[idx:idx_extent]
        # volt[k] = mtag.retrieve_data(k, 0)[:]
        parameters[k] = [k, frequency[k], amplitude[k]]

    # Save Data to HDD
    dname = data_files + 'FIField_voltage'
    dname2 = data_files + 'FIField_parameters.npy'
    np.save(dname2, parameters)
    with open(dname, 'wb') as fp:
        pickle.dump(volt, fp)

    f.close()
    print('Voltage saved')
    return 0


def fifield_spike_detection(path_names, th_factor=4, th_window=400, mph_percent=0.8, filter_on=True, valley=False, min_th=20, save_data=True, show=False):
    # (data_name, dynamic, valley, th_factor=4, min_dist=70, maxph=10, th_window=400, filter_on=True):
    data_name = path_names[0]
    pathname = path_names[1]
    data_file = pathname + 'FIField_voltage'
    parameters_file = pathname + 'FIField_parameters.npy'
    parameters = np.load(parameters_file)
    fs = 100 * 1000
    oo = 0
    with open(data_file, 'rb') as fp:
        volt = pickle.load(fp)

    spike_times = [[]] * len(volt)
    spike_times_valley = [[]] * len(volt)
    early_spikes = [[]] * len(volt)
    stimulus_duration = len(volt[0]) / fs

    for k in tqdm(range(len(volt)), desc='Spike Detection'):
        x = volt[k]
        # Filter Voltage Trace
        if filter_on:
            nyqst = 0.5 * fs
            lowcut = 100
            highcut = 4000
            low = lowcut / nyqst
            high = highcut / nyqst
            x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)
        if valley:
            x = -x

        # Peak Detection using indexes()
        # spike_times[k], thres = indexes(x, dynamic, th_factor, min_dist, maxph, th_window)

        # Thunderfish peakdetection: window size in seconds
        th = pk.std_threshold(x, samplerate=fs, win_size=th_window, th_factor=th_factor)
        if np.min(th) <= min_th and th_window is None:
            th = min_th
        spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

        # Remove large spikes
        spike_times[k], spike_times_valley[k], marked, marked_valley = remove_large_spikes(x, spike_times[k],
                                                                                           spike_times_valley[k],
                                                                                           mph_percent, method='std')
        # Remove early spikes
        early_limit = 0.005 * fs
        early_spikes[k] = spike_times[k][spike_times[k] < early_limit]
        spike_times[k] = spike_times[k][spike_times[k] >= early_limit]


        # Plot Spike Detection
        # if np.random.rand(1) > 0.98:
        # if parameters[k][2] > 70 and np.random.rand(1) > 0.8:
        if show:
            # if parameters[k][1] == 50000 and (parameters[k][2] > 70 or (parameters[k][2] < 45 and parameters[k][2] > 35)):
            # if np.random.rand(1) > 0.90:
            # if k == 20:
            if (parameters[k][1] >= 40000 and parameters[k][1] <= 70000) and parameters[k][2] < 50:  # and oo == 0:
                oo = 1
                plt.figure()
                plt.plot(x)
                plt.xlabel('Time')
                plt.ylabel('Voltage [uV]')
                plt.ylim(-150, 150)
                if spike_times[k].any():
                    plt.plot(spike_times[k], x[spike_times[k]], 'ro')
                if spike_times_valley[k].any():
                    plt.plot(spike_times_valley[k], x[spike_times_valley[k]], 'bo')
                if not np.isnan(marked).all():
                    plt.plot(marked, x[marked], 'kx')
                plt.plot(early_spikes[k], x[early_spikes[k]], 'go')
                if th_window is None:
                    plt.plot([0, len(x)], [th, th], 'r--')
                    plt.plot([0, len(x)], [-th, -th], 'r--')
                    plt.title('ID: {:.0f}'.format(parameters[k][0]) + ', ' + '{:.0f} kHz'.format(
                        parameters[k][1] / 1000) + ', ' + '{:.0f} dB'.format(parameters[k][2]) + ', '
                              + 'th: {:.0f}'.format(th))
                else:
                    plt.plot(th / 2, 'b--')
                    plt.plot(-th / 2, 'b--')
                    plt.title('ID: {:.0f}'.format(parameters[k][0]) + ', ' + '{:.0f} kHz'.format(
                        parameters[k][1] / 1000) + ', ' + '{:.0f} dB'.format(parameters[k][2]))

                plt.ylim(np.min(x)-20, np.max(x)+20)
                plt.show()
        spike_times[k] = spike_times[k] / fs  # now in seconds
        spike_times_valley[k] = spike_times_valley[k] / fs  # now in seconds

    # Save Data to HDD
    if save_data:
        # if valley:
        #     spike_times = spike_times_valley
        dname = pathname + 'FIField_spike_times'
        with open(dname, 'wb') as fp:
            pickle.dump(spike_times, fp)
        dname2 = pathname + 'FIField_duration'
        with open(dname2, 'wb') as fp2:
            pickle.dump(stimulus_duration, fp2)

        print('Spike Times saved')

    return spike_times, spike_times_valley


def fifield_analysis2(path_names):
    data_name = path_names[0]
    pathname = path_names[1]
    data_file = pathname + 'FIField_spike_times'
    parameters_file = pathname + 'FIField_parameters.npy'
    stim_dur = pathname + 'FIField_duration'

    parameters = np.load(parameters_file)
    with open(data_file, 'rb') as fp:
        spike_times = pickle.load(fp)

    with open(stim_dur, 'rb') as fp2:
        stimulus_duration = pickle.load(fp2)

    spike_times = np.array(spike_times)
    parameters[:, 2] = np.floor(parameters[:, 2])  # round strange dB values
    used_freqs = np.unique(parameters[:, 1])
    spike_count = {}
    firing_rate = {}
    first_spike_latency = {}
    first_spike_isi = {}
    d_isi = {}
    instant_firing_rate = {}
    conv_rate = {}
    for i in tqdm(range(len(used_freqs)), desc='FI Analysis'):
        # Get all trials with frequency i
        y = spike_times[parameters[:, 1] == used_freqs[i]]
        amps = parameters[parameters[:, 1] == used_freqs[i], 2]
        used_amps = np.unique(amps)
        spc = np.zeros((len(used_amps), 3))
        rate = np.zeros((len(used_amps), 3))
        fsl = np.zeros((len(used_amps), 3))
        fisi = np.zeros((len(used_amps), 3))
        d = np.zeros((len(used_amps), 4))
        instant_rate = np.zeros((len(used_amps), 3))
        c_rate = np.zeros((len(used_amps), 3))
        for k in range(len(used_amps)):
            # Get all trials with amplitude k
            x = y[amps == used_amps[k]]
            if sum(amps == used_amps[k]) == 0:
                continue

            # Spike Train Distance
            # no_empty_trial = False
            # for g in range(len(x)):
            #     if len(x[g]) == 0:
            #         no_empty_trial = True
            #         break

            if len(x[0]) > 0:
                edges = [0, stimulus_duration]
                trains = [[]] * len(x)
                for p in range(len(x)):
                    trains[p] = spk.SpikeTrain(x[p], edges)
                d[k, 1] = spk.isi_distance(trains, interval=edges)
                d[k, 2] = spk.spike_sync(trains, interval=edges)
                d[k, 3] = spk.spike_distance(trains, interval=edges)
            else:
                d[k, 1] = np.nan

            # Rate and First Spike Statistics
            single_count = np.zeros((len(x), 1))
            single_rate = np.zeros((len(x), 1))
            first_spike = np.zeros((len(x), 1))
            first_isi = np.zeros((len(x), 1))
            i_rate = np.zeros((len(x), 1))
            for j in range(len(x)):  # Loop over all single trials and count spikes
                single_count[j] = len(x[j])  # Count spikes per trial
                single_rate[j] = len(x[j]) / stimulus_duration

                if len(x[j]) <= 1:
                    first_spike[j] = np.nan
                    first_isi[j] = np.nan
                    i_rate[j] = np.nan
                else:
                    first_spike[j] = x[j][0] * 1000  # First Spike Latency
                    first_isi[j] = (x[j][1] - x[j][0]) * 1000  # Fist Spike ISI in ms
                    i_rate[j] = np.mean(1/np.diff(x[j]))
            spc[k, 2] = np.std(single_count)
            spc[k, 1] = np.mean(single_count)
            spc[k, 0] = used_amps[k]
            rate[k, 2] = np.std(single_rate)
            rate[k, 1] = np.mean(single_rate)
            rate[k, 0] = used_amps[k]
            d[k, 0] = used_amps[k]
            instant_rate[k, 0] = used_amps[k]
            dt = 1 / (100*1000)
            sigma = 0.005
            _, _, _, rr, rr_std = convolution_rate(x, t_max=stimulus_duration, dt=dt, sigma=sigma)
            c_rate[k, 0] = used_amps[k]
            c_rate[k, 1] = rr
            c_rate[k, 2] = rr_std

            if np.isnan(i_rate).all():
                instant_rate[k, 1] = np.nan
                instant_rate[k, 2] = np.nan
            else:
                instant_rate[k, 1] = np.nanmean(i_rate)
                instant_rate[k, 2] = np.nanstd(i_rate)

            if np.isnan(first_spike).all():
                fsl[k, 2] = np.nan
                fsl[k, 1] = np.nan
                fisi[k, 2] = np.nan
                fisi[k, 1] = np.nan
            else:
                fsl[k, 2] = np.nanstd(first_spike)
                fsl[k, 1] = np.nanmean(first_spike)
                fisi[k, 2] = np.nanstd(first_isi)
                fisi[k, 1] = np.nanmean(first_isi)
            fsl[k, 0] = used_amps[k]
            fisi[k, 0] = used_amps[k]
        spike_count.update({used_freqs[i]/1000: spc})
        firing_rate.update({used_freqs[i] / 1000: rate})
        first_spike_latency.update({used_freqs[i]/1000: fsl})
        first_spike_isi.update({used_freqs[i] / 1000: fisi})
        d_isi.update({used_freqs[i] / 1000: d})
        instant_firing_rate.update({used_freqs[i] / 1000: instant_rate})
        conv_rate.update({used_freqs[i] / 1000: c_rate})

    return spike_count, firing_rate, first_spike_latency, first_spike_isi, d_isi, instant_firing_rate, conv_rate, stimulus_duration


# ----------------------------------------------------------------------------------------------------------------------
# MOTH AND BAT CALLS:

def reconstruct_moth_song(meta):
    # Get metadata
    # samplingrate = np.round(meta[1]) * 1000  # in Hz
    samplingrate = 800 * 1000  # This is equals speaker output sampling rate
    pulsenumber = len(meta.sections)
    MetaData = np.zeros((5, pulsenumber))

    # Reconstruct Moth Song Stimulus from Metadata
    for i in range(pulsenumber):
        a = meta.sections[i]
        MetaData[0, i] = a["Duration"]
        MetaData[1, i] = a["Tau"]
        MetaData[2, i] = a["Frequency"]
        MetaData[3, i] = a["StartTime"]
        MetaData[4, i] = a["Amplitude"]
    stimulus_duration = MetaData[0, -1] + MetaData[3, -1] + 0.01  # in secs
    stimulus = np.zeros(int(samplingrate * stimulus_duration))  # This must also be read from metadata!
    stimulus_time = np.linspace(0, stimulus_duration, samplingrate * stimulus_duration)
    pulse = np.zeros((int(pulsenumber), int(samplingrate * MetaData[0, 0] + 1)))
    pulse_time = np.linspace(0, MetaData[0, 0], samplingrate * MetaData[0, 0])  # All pulses must have same length!
    for p in range(pulsenumber):
        j = 0
        for t in pulse_time:
            pulse[p, j] = np.sin(2 * np.pi * MetaData[2, p] * t) * np.exp(-t / MetaData[1, p])
            j += 1
        p0 = int(MetaData[3, p] * samplingrate)
        p1 = int(p0 + (MetaData[0, p] * samplingrate))
        stimulus[p0:p1 + 1] = pulse[p, :]

    gap = MetaData[3, 3] - MetaData[3, 2]  # pause between single pulses
    return stimulus_time, stimulus, gap, MetaData


def get_soundfilestimuli_data(datasets, tag, plot):
    data_name = datasets[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + data_name + '.nix'
    # tag = 'SingleStimulus-file-5'

    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]

    # Get tags
    mtag = b.multi_tags[tag]

    # Get meta data
    meta = mtag.metadata.sections[0]

    # Get sound file: 0: sampling rate, 1: sound data
    sound_file_name = meta[2]
    sound = wav.read('/media/brehm/Data/MasterMoth/stimuli/' + sound_file_name)
    sound_time = np.linspace(0, len(sound[1]) / sound[0], len(sound[1])) * 1000  # in ms

    # Read Voltage Traces from nix file
    sampling_rate = 100 * 1000  # in Hz
    voltage = {}
    voltage_time = {}
    trials = 5
    # recording_length = mtag.retrieve_data(0, 0)[:].shape
    # voltage = np.zeros((trials, recording_length[0]))
    for k in range(trials):
        # voltage.update({k: mtag.retrieve_data(k, 0)[:]})  # Voltage Trace
        v = mtag.retrieve_data(k, 0)[:]
        fs = 100 * 1000
        nyqst = 0.5 * fs
        lowcut = 200
        highcut = 4000
        low = lowcut / nyqst
        high = highcut / nyqst
        # voltage[k, :] = voltage_trace_filter(v, [low, high], ftype='band', order=3, filter_on=True)
        voltage.update({k: voltage_trace_filter(v, [low, high], ftype='band', order=2, filter_on=True)})
        voltage_time.update({k: np.linspace(0, len(voltage[k]) / sampling_rate, len(voltage[k])) * 1000})  # in ms

    # voltage = voltage / np.max(voltage)
    s = sound[1]
    # s = s / np.max(s)

    # Plot sound stimulus and voltage trace
    if plot:
        for j in range(trials):
            plt.subplot(trials+1, 1, j+1)
            plt.plot(voltage_time[j], voltage[j], 'k')
            if j == 0:
                plt.title('Stimulus: %s' % sound_file_name)


        plt.subplot(trials+1, 1, trials+1)
        plt.plot(sound_time, s, 'k')
        plt.xlabel('time [ms]')
        plt.show()

    # Save Data to HDD
    dname = pathname + sound_file_name[:-4] + '_voltage.npy'
    np.save(dname, voltage)
    dname2 = pathname + sound_file_name[:-4] + '_time.npy'
    np.save(dname2, voltage_time)
    print('%s done' % sound_file_name)

    f.close()
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# SPIKE TRAIN DISTANCE

def fast_e_pulses(spikes, tau, dt):
    # This is the new and much faster method
    def exp_kernel(ta, step):
        x = np.arange(0, tau*5, step)
        y = np.exp(-x/ta)
        return y
    # dt = 0.001
    t_max = spikes[-1] + 5 * tau

    spike_times = spikes[spikes < t_max]
    t = np.arange(0, t_max - dt, dt)
    r = np.zeros(len(t))
    spike_ids = np.round(spike_times / dt) - 2
    r[spike_ids.astype('int')] = 1
    kernel = exp_kernel(tau, dt)
    pulses = np.convolve(r, kernel, mode='full')

    return pulses


def vanrossum_distance(f, g, dt, tau):
    # Make sure both trains have same length
    difference_in_length = abs(len(f) - len(g))
    if len(f) > len(g):
        g = np.append(g, np.zeros(difference_in_length))
    elif len(f) < len(g):
        f = np.append(f, np.zeros(difference_in_length))

    # Compute VanRossum Distance

    d = np.sum((f - g) ** 2) * (dt / tau)
    # f_count = np.sum(f)* (dt / tau)
    # g_count = np.sum(g) * (dt / tau)
    return d


def trains_to_e_pulses(path_names, tau, dt, stim_type, method):
    dataset = path_names[0]
    pathname = path_names[1]
    spikes = np.load(pathname + 'Calls_spikes.npy').item()

    if stim_type == 'selected':
        stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']

    if stim_type == 'series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'all_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'moth_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_series_selected':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav']

    if stim_type == 'moth_single_selected':
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

    if stim_type == 'all_single':
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
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_single':
        stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_series':
        stims = ['callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    # Tags and Stimulus names
    connection, _ = tagtostimulus(path_names)
    stimulus_tags = [''] * len(stims)
    for p in range(len(stims)):
        stimulus_tags[p] = connection[stims[p]]

    # Convert all Spike Trains to e-pulses
    trains = {}
    for k in tqdm(range(len(stimulus_tags)), desc='Converting pulses'):
        trial_nr = len(spikes[stimulus_tags[k]])
        tr = [[]] * trial_nr
        for j in range(trial_nr):
            x = spikes[stimulus_tags[k]][j]
            # tr[j] = spike_e_pulses(x, dt_factor, tau, duration, whole_train, method)
            tr[j] = fast_e_pulses(x, tau, dt)
        trains.update({stimulus_tags[k]: tr})

    # Save to HDD
    if method == 'rect':
        file_name = pathname + 'rect_trains_' + str(int(tau * 1000)) + '_' + stim_type + '.npy'
        file_name2 = pathname + 'rect_stimulus_tags_' + str(int(tau * 1000)) + '_' + stim_type + '.npy'
    else:
        file_name = pathname + 'e_trains_' + str(int(tau*1000)) + '_' + stim_type + '.npy'
        file_name2 = pathname + 'stimulus_tags_' + str(int(tau * 1000)) + '_' + stim_type + '.npy'

    np.save(file_name, trains)
    np.save(file_name2, stimulus_tags)
    return 0


def pulse_trains_to_e_pulses(samples, tau, dt):
    e_pulses = [[]] * len(samples)
    for k in range(len(samples)):
        # e_pulses[k] = spike_e_pulses(np.array(samples[k]), dt_factor, tau, duration, whole_train, method)
        e_pulses[k] = fast_e_pulses(np.array(samples[k]), tau, dt)

    return e_pulses


def vanrossum_matrix(dataset, trains, stimulus_tags, duration, dt, tau, boot_sample, save_fig):

    e_pulse_fs = dt
    duration_in_samples = int(duration / e_pulse_fs)
    call_count = len(stimulus_tags)

    # Select Template and Probes and bootstrap this process
    mm = {}
    gg = {}
    distances_per_boot = {}
    for boot in range(boot_sample):
        count = 0
        match_matrix = np.zeros((call_count, call_count))
        group_matrix = np.zeros((2, call_count))
        templates = {}
        probes = {}
        # rand_ids = np.random.randint(trial_nr, size=call_count)
        for i in range(call_count):
            trial_nr = len(trains[stimulus_tags[i]])
            rand_id = np.random.randint(trial_nr, size=1)
            idx = np.arange(0, trial_nr, 1)
            # idx = np.delete(idx, rand_ids[i])
            # templates.update({i: trains[stimulus_tags[i]][rand_ids[i]]})
            idx = np.delete(idx, rand_id[0])

            # Get Templates
            templates.update({i: trains[stimulus_tags[i]][rand_id[0]][0:duration_in_samples]})

            # Get Probes
            for q in range(len(idx)):
                probes.update({count: [trains[stimulus_tags[i]][idx[q]][0:duration_in_samples], i]})
                count += 1

        # Compute VanRossum Distance
        distances = [[]] * len(probes)
        call_ids = [[]] * len(probes)
        for pr in range(len(probes)):
            d = np.zeros(len(templates))
            for tmp in range(len(templates)):
                d[tmp] = vanrossum_distance(templates[tmp], probes[pr][0], dt, tau)

            template_match = np.where(d == np.min(d))[0][0]
            song_id = probes[pr][1]
            match_matrix[template_match, song_id] += 1
            distances[pr] = d
            call_ids[pr] = probes[pr][1]

            # 0: Moths, 1: Bats
            probe_group = int(float(stimulus_tags[song_id][20:]) >= 82)
            template_group = int(float(stimulus_tags[template_match][20:]) >= 82)
            group_matrix[template_group, song_id] += 1

        mm.update({boot: match_matrix})
        gg.update({boot: group_matrix})
        md = [[]] * len(np.unique(call_ids))
        distances = np.array(distances)
        for qq in range(len(np.unique(call_ids))):
            idx = call_ids == np.unique(call_ids)[qq]
            md[qq] = np.mean(distances[idx], 0)
        md = np.array(md)
        distances_per_boot.update({boot: [md, call_ids]})

    mm_mean = sum(mm.values()) / len(mm)
    gg_mean = sum(gg.values()) / len(gg)

    # Percent Correct
    percent_correct = np.zeros((len(mm_mean)))
    correct_nr = np.zeros((len(mm_mean)))
    for r in range(len(mm_mean)):
        percent_correct[r] = mm_mean[r, r] / np.sum(mm_mean[:, r])
        correct_nr[r] = mm_mean[r, r]

    correct_matches = np.sum(correct_nr) / np.sum(mm_mean)

    if save_fig:
        # Plot Matrix
        plot_settings()
        plt.imshow(mm_mean)
        plt.xlabel('Original Calls')
        plt.ylabel('Matched Calls')
        plt.colorbar()
        plt.xticks(np.arange(0, len(mm_mean), 1))
        plt.yticks(np.arange(0, len(mm_mean), 1))
        plt.title('tau = ' + str(tau*1000) + ' ms' + ' T = ' + str(duration*1000) + ' ms')

        # Save Plot to HDD
        figname = "/media/brehm/Data/MasterMoth/figs/" + dataset + '/VanRossumMatrix_' + str(tau*1000) + \
                  '_' + str(duration*1000) + '.png'
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # print(figname)
        # print('tau = ' + str(tau*1000) + ' ms' + ' T = ' + str(duration*1000) + ' ms done')

    return mm_mean, correct_matches, distances_per_boot, gg_mean


def isi_matrix(path_names, duration, boot_sample, stim_type, profile, save_fig):
    dataset = path_names[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    spikes = np.load(pathname + 'Calls_spikes.npy').item()
    # tag_list = np.load(pathname + 'Calls_tag_list.npy')
    '''
    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
             'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
             'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']


    stims = ['naturalmothcalls/BCI1062_07x07.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav',
             'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/neritos_cotes_07x07.wav']


    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav',
             'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav']


    stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                'batcalls/Eptesicus_nilssonii_1_s.wav',
                'batcalls/Myotis_bechsteinii_1_n.wav',
                'batcalls/Myotis_brandtii_1_n.wav',
                'batcalls/Myotis_nattereri_1_n.wav',
                'batcalls/Nyctalus_leisleri_1_n.wav',
                'batcalls/Nyctalus_noctula_2_s.wav',
                'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                'batcalls/Vespertilio_murinus_1_s.wav']

    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
             'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
             'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav',
             'batcalls/Barbastella_barbastellus_1_n.wav',
             'batcalls/Eptesicus_nilssonii_1_s.wav',
             'batcalls/Myotis_bechsteinii_1_n.wav',
             'batcalls/Myotis_brandtii_1_n.wav',
             'batcalls/Myotis_nattereri_1_n.wav',
             'batcalls/Nyctalus_leisleri_1_n.wav',
             'batcalls/Nyctalus_noctula_2_s.wav',
             'batcalls/Pipistrellus_pipistrellus_1_n.wav',
             'batcalls/Pipistrellus_pygmaeus_2_n.wav',
             'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
             'batcalls/Vespertilio_murinus_1_s.wav']
    '''

    if stim_type == 'selected':
        stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']

    if stim_type == 'series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'all_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'moth_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_series_selected':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav']

    if stim_type == 'moth_single_selected':
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

    if stim_type == 'all_single':
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
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_single':
        stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_series':
        stims = ['callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    # Tags and Stimulus names
    connection, _ = tagtostimulus(path_names)
    stimulus_tags = [''] * len(stims)
    for p in range(len(stims)):
        stimulus_tags[p] = connection[stims[p]]

    # trial_nr = 20
    '''
    ln = np.zeros((len(stimulus_tags)))
    for k in range(len(stimulus_tags)):
        ln[k] = len(spikes[stimulus_tags[k]])
    trial_nr = np.min(ln)
    '''
    call_count = len(stimulus_tags)

    # Select Template and Probes and bootstrap this process
    mm = {}
    distances_per_boot = {}
    rand = {}
    for boot in range(boot_sample):
        count = 0
        match_matrix = np.zeros((call_count, call_count))
        random_matrix = np.zeros((call_count, call_count))
        templates = {}
        probes = {}
        # rand_ids = np.random.randint(trial_nr, size=call_count)
        for i in range(call_count):
            trial_nr = len(spikes[stimulus_tags[i]])
            rand_id = np.random.randint(trial_nr, size=1)
            idx = np.arange(0, trial_nr, 1)
            # idx = np.delete(idx, rand_ids[i])
            # templates.update({i: spikes[stimulus_tags[i]][rand_ids[i]]})
            idx = np.delete(idx, rand_id[0])
            templates.update({i: spikes[stimulus_tags[i]][rand_id[0]]})
            for q in range(len(idx)):
                probes.update({count: [spikes[stimulus_tags[i]][idx[q]], i]})
                count += 1

        # Compute ISI Distance
        distances = [[]] * len(probes)
        call_ids = [[]] * len(probes)
        for pr in range(len(probes)):
            d = np.zeros(len(templates))
            for tmp in range(len(templates)):
                edges = [0, duration]
                temp = spk.SpikeTrain(templates[tmp], edges)
                prb = spk.SpikeTrain(probes[pr][0], edges)
                if profile == 'COUNT':
                    profile_name = '/COUNT_Matrix_'
                    d[tmp] = abs(len(prb.spikes[prb.spikes <= duration])-len(temp.spikes[temp.spikes <= duration]))

                if profile == 'ISI':
                    # ISI Profile:
                    profile_name = '/ISI_Matrix_'
                    d[tmp] = spk.isi_distance(temp, prb, interval=[0, duration])
                    # isi_profile = spk.isi_profile(temp, prb)
                    # d[tmp] = isi_profile.avrg()

                if profile == 'SPIKE':
                    # SPIKE Profile:
                    profile_name = '/SPIKE_Matrix_'
                    d[tmp] = spk.spike_distance(temp, prb, interval=[0, duration])
                    # SPIKE_profile = spk.spike_profile(temp, prb)
                    # d[tmp] = SPIKE_profile.avrg()

                if profile == 'SYNC':
                    profile_name = '/SYNC_Matrix_'
                    d[tmp] = spk.spike_sync(temp, prb, interval=[0, duration])

                if profile == 'DUR':
                    profile_name = '/DUR_Matrix'
                    # d[tmp] = abs(temp[-1] - prb[-1])
                    if prb.spikes[prb.spikes <= duration].any():
                        prb_dur = prb.spikes[prb.spikes <= duration][-1]  # time point of last spike
                    else:
                        prb_dur = 0

                    if temp.spikes[temp.spikes <= duration].any():
                        temp_dur = temp.spikes[temp.spikes <= duration][-1]
                    else:
                        temp_dur = 0
                    d[tmp] = abs(prb_dur - temp_dur)

            if profile == 'SYNC':
                template_match = np.where(d == np.max(d))[0][0]
            else:
                template_match = np.where(d == np.min(d))[0][0]

            song_id = probes[pr][1]
            match_matrix[template_match, song_id] += 1
            random_matrix[np.random.randint(0, call_count-1, 1)[0], song_id] += 1
            distances[pr] = d
            call_ids[pr] = probes[pr][1]
            '''
            if len(np.where(d == np.min(d))) > 1:
                print('Found more than 1 min.')
            if pr == 0:
                print(d)
                print('---------------')

                plt.figure()
                plt.subplot(3, 1, 1)
                plt.plot(templates[template_match])
                plt.title('Matched Template: ' + str(d[template_match]))
                plt.subplot(3, 1, 2)
                plt.plot(templates[song_id])
                plt.title('Probes correct Template: ' + str(d[song_id]))
                plt.subplot(3, 1, 3)
                plt.plot(probes[pr][0])
                plt.title('Probe')
                plt.show()
                embed()
            '''
        mm.update({boot: match_matrix})
        rand.update({boot: random_matrix})
        md = [[]] * len(np.unique(call_ids))
        distances = np.array(distances)
        for qq in range(len(np.unique(call_ids))):
            idx = call_ids == np.unique(call_ids)[qq]
            md[qq] = np.mean(distances[idx], 0)
        md = np.array(md)
        distances_per_boot.update({boot: [md, call_ids]})

    mm_mean = sum(mm.values()) / len(mm)
    rand_mean = sum(rand.values()) / len(rand)

    # Percent Correct
    percent_correct = np.zeros((len(mm_mean)))
    correct_nr = np.zeros((len(mm_mean)))
    rand_correct_nr = np.zeros((len(rand_mean)))
    for r in range(len(mm_mean)):
        percent_correct[r] = mm_mean[r, r]/np.sum(mm_mean[:, r])
        correct_nr[r] = mm_mean[r, r]
        rand_correct_nr[r] = rand_mean[r, r]

    correct_matches = np.sum(correct_nr)/np.sum(mm_mean)
    rand_correct_matches = np.sum(rand_correct_nr)/np.sum(rand_mean)

    if save_fig:
        # Plot Matrix
        plt.imshow(mm_mean)
        plt.xlabel('Original Calls')
        plt.ylabel('Matched Calls')
        plt.colorbar()
        plt.xticks(np.arange(0, len(mm_mean), 1))
        plt.yticks(np.arange(0, len(mm_mean), 1))
        plt.title('T = ' + str(duration * 1000) + ' ms')

        # Save Plot to HDD
        figname = "/media/brehm/Data/MasterMoth/figs/" + dataset + profile_name + str(duration * 1000) + '.png'
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        fig.savefig(figname, bbox_inches='tight', dpi=100)
        plt.close(fig)
        # print('T = ' + str(duration * 1000) + ' ms done')

    return mm_mean, correct_matches, distances_per_boot, rand_correct_matches


def pulse_train_matrix(samples, duration, profile):
    d = np.zeros((len(samples), len(samples)))
    edges = [0, duration]
    for i in range(len(samples)):
        # template = samples[i]
        template = spk.SpikeTrain(samples[i], edges)
        for k in range(len(samples)):
            probe = spk.SpikeTrain(samples[k], edges)
            if profile == 'COUNT':
                d[i, k] = abs(len(probe.spikes[probe.spikes <= duration])-len(template.spikes[template.spikes <= duration]))
            if profile == 'ISI':
                d[i, k] = spk.isi_distance(probe, template, interval=[0, duration])
            if profile == 'SPIKE':
                d[i, k] = spk.spike_distance(probe, template, interval=[0, duration])
            if profile == 'SYNC':
                d[i, k] = spk.spike_sync(probe, template, interval=[0, duration])
            if profile == 'DUR':
                if probe.spikes[probe.spikes <= duration].any():
                    prb_dur = probe.spikes[probe.spikes <= duration][-1]
                else:
                    prb_dur = 0

                if template.spikes[template.spikes <= duration].any():
                    temp_dur = template.spikes[template.spikes <= duration][-1]
                else:
                    temp_dur = 0
                d[i, k] = abs(prb_dur - temp_dur)
    return d

# ----------------------------------------------------------------------------------------------------------------------
# Interval MothASongs functions (MAS):


def get_moth_intervals_data(path_names, save_data):
    # Get voltage trace from nix file for Intervals MothASongs Protocol
    # data_set_numbers has to fit the the data names in the nix file!
    # Returns voltage trace, stimulus time, stimulus amplitude, gap and meta data for every single trial
    # datasets can be a list of many recordings

    data_name = path_names[0]
    pathname = path_names[1]
    nix_file = path_names[3] + '.nix'

    # Open the nix file
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]

    # Find all multi tags:
    m = 'MothASongs-moth_song-damped'
    mtag_list = [t.name for t in b.multi_tags if m in t.name]  # Find Multi-Tags

    # Set Data Parameters
    skips = 0

    # Be careful not to load data that is not from the interval protocol
    # data_set_numbers = ['351-1','176-1','117-1','88-1','71-1','59-1','51-1','44-1','39-1','36-1','32-1','30-1',
    #                     '27-1','26-1','24-1','117-2','88-2','71-2','59-2','51-2','44-2','39-2','36-2','32-2',
    #                     '30-2','27-2','26-2','24-2','22-1','21-1']

    voltage = {}
    for sets in tqdm(range(len(mtag_list)), desc='Artificial Moth Intervals'):

        mtag = b.multi_tags[mtag_list[sets]]
        trial_number = len(mtag.positions[:])
        if trial_number <= 1:  # if only one trial exists, skip this data
            skips += 1
            print('skipped %s: less than 2 trials found' % str(sets))
            continue

        # Get metadata
        meta = mtag.metadata.sections[0]

        # Reconstruct stimulus from meta data
        stimulus_time, stimulus, gap, MetaData = reconstruct_moth_song(meta)
        tau = MetaData[1, 0]
        frequency = MetaData[2, 0]
        freq = int(frequency / 1000)

        # Get voltage trace
        volt = {}
        for trial in range(trial_number):
            v = mtag.retrieve_data(trial, 0)[:]
            volt.update({trial: {0: v, 1: tau, 2: gap, 3: freq}})
        volt.update({'stimulus': stimulus, 'stimulus_time': stimulus_time, 'gap': gap, 'meta': MetaData, 'trials': trial_number})
        voltage.update({sets: volt})

    if save_data:
        # Save data to HDD
        dname = pathname + 'intervals_mas_voltage.npy'
        np.save(dname, voltage)
        # dname2 = pathname + 'intervals_mas_stim_time.npy'
        # np.save(dname2, stimulus_time)
        # dname3 = pathname + 'intervals_mas_stim.npy'
        # np.save(dname3, stimulus)
        # dname4 = pathname + 'intervals_mas_gap.npy'
        # np.save(dname4, gap)
        # dname5 = pathname + 'intervals_mas_metadata.npy'
        # np.save(dname5, MetaData)

    f.close()

    return voltage


def moth_intervals_spike_detection(path_names, window=None, th_factor=1, mph_percent=0.8, filter_on=True,
                                   save_data=True, show=[False, False, False], bin_size=0.01):
    # Load data
    fs = 100 * 1000
    data_name = path_names[0]
    pathname = path_names[1]
    fname = pathname + 'intervals_mas_voltage.npy'
    voltage = np.load(fname).item()

    # Possion Spikes
    nsamples = 1000
    tmax = 0.35
    tmin = 0.005
    p_spikes, isi_p = poission_spikes(nsamples, 100, tmax)
    pp = np.arange(0.0005, 0.05, 0.0005)
    vs_boot, phase_boot, vs_mean_boot, vs_std_boot, vs_percentile_boot, peaks_boot = \
        vs_range(p_spikes, pp, tmin=tmin)

    # Now detect spikes in each trial and update input data
    for i in voltage:
        spikes = []
        stimulus = voltage[i]['stimulus']
        stimulus_time = voltage[i]['stimulus_time']
        info = str(voltage[i]['gap']*1000)
        trial_number = voltage[i]['trials']
        spike_times = [[]] * trial_number
        spike_times_valley = [[]] * trial_number

        for k in range(trial_number):
            x = voltage[i][k][0]
            # Filter Voltage Trace
            if filter_on:
                nyqst = 0.5 * fs
                lowcut = 300
                highcut = 2000
                low = lowcut / nyqst
                high = highcut / nyqst
                x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)

            th = pk.std_threshold(x, fs, window, th_factor)
            spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

            # Remove large spikes
            t = np.arange(0, len(x) / fs, 1 / fs)
            spike_size = pk.peak_size_width(t, x, spike_times[k], spike_times_valley[k], pfac=0.75)
            spike_times[k], spike_times_valley[k], marked, marked_valley = \
                remove_large_spikes(x, spike_times[k], spike_times_valley[k], mph_percent=mph_percent, method='std')

            # Plot Spike Detection
            # Cut out spikes
            fs = 100 * 1000
            snippets = pk.snippets(x, spike_times[k], start=-100, stop=100)
            snippets_removed = pk.snippets(x, marked, start=-100, stop=100)

            if show[0] and k == 0:
                plot_spike_detection_gaps(x, spike_times[k], spike_times_valley[k], marked, spike_size, mph_percent,
                                          snippets, snippets_removed, th, window, info, stimulus_time, stimulus)

            spike_times[k] = spike_times[k] / fs  # Now spike times are in real time (seconds)
            spike_times_valley[k] = spike_times_valley[k] / fs
            spikes = np.append(spikes, spike_times[k])  # Put spike times of each trial in one long array
            spike_count = len(spike_times[k])
            voltage[i][k].update({'spike_times': spike_times[k], 'spike_count': spike_count})
        voltage[i].update({'all_spike_times': spikes})

        # Get mean firing rate over all trials
        f_rate, b = psth(spike_times, bin_size, plot=False, return_values=True, tmax=tmax, tmin=tmin)
        mean_rate = np.mean(f_rate)
        std_rate = np.std(f_rate)

        # VS Range
        vs, phase, vs_mean, vs_std, vs_percentile, peaks = vs_range(spike_times, pp, tmin=tmin)

        # Poisson Boot
        # Project spike times onto circle and get the phase (angle)
        t_period = voltage[i]['gap']
        vector_boot = np.exp(np.dot(2j * np.pi / t_period, np.concatenate(p_spikes)))
        vector_phase_boot = np.angle(vector_boot)

        # Rayleigh Test
        p_boot_rayleigh, z_boot_rayleigh = c_stat.rayleigh(vector_phase_boot)

        # Project spike times onto circle and get the phase (angle)
        # v = np.exp(2j * np.pi * spike_times/t_period)
        vector = np.exp(np.dot(2j * np.pi / t_period, np.concatenate(spike_times)))
        vector_phase = np.angle(vector)

        # Rayleigh Test
        p_rayleigh, z_rayleigh = c_stat.rayleigh(vector_phase)
        # V-Test
        mu = np.mean(vector)
        p_vtest, z_vtest = c_stat.vtest(vector_phase, np.angle(mu))

        print('gap = period (ms): ' + str(t_period) + ', p = ' + str(np.round(p_rayleigh, 4)) + ', z = ' + str(
            z_rayleigh))
        print('Poisson Boot: ' + ' p = ' + str(np.round(p_boot_rayleigh, 4)) + ', z = ' + str(z_boot_rayleigh))
        print('V-Test: p = ' + str(p_vtest) + ', z = ' + str(z_vtest))
        print('')

        # ISI Histogram and VS vs period
        if show[2] is True:
            fig = plt.figure(figsize=(14, 4))
            # ISI Histogram
            plt.subplot(1, 3, 1)
            bs = 1  # in ms
            plot_isi_histogram(spike_times, p_spikes, bs, x_limit=50, steps=5, info=[False, False],
                               intervals=isi_p, tmin=tmin, plot_title=False)

            # VS vs period
            plt.subplot(1, 3, 2)
            plt.plot(pp * 1000, vs_mean_boot, 'g')
            plt.plot(pp * 1000, vs_percentile_boot, 'r--')
            plt.plot(pp * 1000, vs_std_boot + vs_mean_boot, 'g--')
            if peaks.any():
                plt.plot(pp[peaks] * 1000, vs_mean[peaks], 'ro')
            plt.plot([t_period*1000, t_period*1000], [0, np.max(vs_mean)], 'bx--')
            plt.plot(pp * 1000, vs_mean, 'k')
            plt.fill_between(pp * 1000, vs_mean - vs_std, vs_mean + vs_std, facecolor='k', alpha=0.5)
            plt.xlabel('period [ms]')
            plt.ylabel('Mean VS')
            plt.title('period = ' + str(t_period*1000) + ', f_rate = ' + str(np.round(mean_rate)) + ' Hz')

            # Polar Hist
            ax = plt.subplot(1, 3, 3, projection='polar')
            ax.hist(vector_phase_boot, bins='auto', density=True, facecolor='k', alpha=0.5)
            ax.hist(vector_phase, bins='auto', density=True, facecolor='r', alpha=0.5)
            # ax.plot(np.angle(mu), np.abs(mu), 'ro')
            ax.arrow(0, 0, np.angle(mu), np.abs(mu), head_starts_at_zero=True, color='r')
            ax.set_ylim(0, 1)
            ax.set_title('period = ' + str(t_period*1000) + ' ms')

        # Plot overview of all trials for this stimulus
        if show[1] is True:
            info = [str(t_period), str(t_period)]
            plot_gaps(x, spike_times, spike_times_valley, marked, spike_size, mph_percent, snippets, snippets_removed, th,
                      window, info, stimulus_time, stimulus, bin_size)
        plt.show()
    # Save detected Spikes to HDD
    if save_data:
        dname = pathname + 'intervals_mas_spike_times.npy'
        np.save(dname, voltage)

    print('Spike Detection done')
    return 0


def moth_intervals_analysis(datasets):
    # Load data
    data_name = datasets
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/DataFiles/"
    fname = pathname + 'intervals_mas_spike_times.npy'
    voltage = np.load(fname).item()
    vector_strength = {}
    for i in voltage:
        try:
            gap = voltage[i]['gap']
            tau = voltage[i][0][1]
            spike_times_sorted = np.sort(voltage[i]['all_spike_times'])   # get spike times of all trials
            vs, phase = sg.vectorstrength(spike_times_sorted, gap)  # all in s
            vector_strength.update({i: {'vs': vs, 'vs_phase': phase, 'gap': gap, 'tau': tau}})
            # voltage[i].update({'vs': vs, 'vs_phase': phase})
        except KeyError:
            print('Key Error')

    # Save VS to HDD
    dname = pathname + 'intervals_mas_vs.npy'
    np.save(dname, vector_strength)
    print('Moth Intervals Analysis done')
    return 0


def moth_intervals_plot(data_name, trial_number, frequency):
    # Load data
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/DataFiles/"
    fname = pathname + 'intervals_mas_spike_times.npy'
    voltage = np.load(fname).item()
    for i in voltage:
        # Get Stimulus
        stimulus = voltage[i]['stimulus']
        stimulus_time = voltage[i]['stimulus_time']
        gap = voltage[i]['gap']
        tau = voltage[i][0][1]

        # Compute PSTH (binned)
        spt = voltage[i]['all_spike_times']
        bin_width = 2  # in ms
        # bins = int(np.round(stimulus_time[-1] / (bin_width / 1000)))  # in secs
        frate, bin_edges = psth(spt, trial_number, bin_width/1000, plot=False, return_values=True, separate_trials=False)

        # Raster Plot
        plt.subplot(3, 1, 1)
        for k in range(trial_number):
            sp_times = voltage[i][k]['spike_times']
            plt.plot(sp_times, np.ones(len(sp_times)) + k, 'k|')

        plt.xlim(0, stimulus_time[-1])
        plt.ylabel('Trial Number')
        plt.title('gap = ' + str(gap * 1000) + ' ms, tau = ' + str(tau * 1000) + ' ms, bin = ' + str(bin_width) +
                  ' ms, freq = ' + str(frequency) + ' kHz')

        # PSTH
        plt.subplot(3, 1, 2)
        plt.plot(bin_edges[:-1], frate, 'k')
        plt.xlim(0, stimulus_time[-1])
        plt.ylabel('Firing Rate [Spikes/s]')

        # Stimulus
        plt.subplot(3, 1, 3)
        plt.plot(stimulus_time, stimulus, 'k')
        plt.xlim(0, stimulus_time[-1])
        plt.yticks([-1, 0, 1])
        # plt.text(1, 0.5, 'gap: ' + str(gap*1000) + ' ms')
        # plt.text(1, 0.3, 'tau: ' + str(tau*1000) + ' ms')
        plt.ylabel('Amplitude')

        plt.xlabel('time [ms]')

        # Save Plot to HDD
        figname = pathname + "MothIntervals_" + str(i) + '.png'
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Progress Bar
        percent = np.round(i / len(voltage), 2)
        print('-- MothInterval Plots: %s %%  --' % (percent * 100))

    print('MothInterval Plots saved!')
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# Rect Intervals

def rect_intervals_plot(data_name):
    # Load data
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    fname = pathname + 'intervals_rect_trials.npy'
    spike_times = np.load(fname).item()
    stimulus = np.load(pathname + 'intervals_rect_stimulus.npy').item()
    trial_number = 10

    for i in spike_times:
        # Get Stimulus
        stimulus_amplitude = stimulus[i]['stimulus']
        stimulus_time = stimulus[i]['stimulus_time']
        gap = stimulus[i]['gap']
        pulse_duration = stimulus[i]['pulse_duration']

        # Compute PSTH (binned)
        spt = []
        for p in spike_times[i]['trials']:
            spt = np.append(spt, spike_times[i]['trials'][p])

        bin_width = 2  # in ms
        bins = int(np.round(stimulus_time[-1] / (bin_width / 1000)))  # in secs
        frate, bin_edges, bin_width2 = psth(spt, trial_number, bins, plot=False, return_values=True)

        # Raster Plot
        plt.subplot(3, 1, 1)
        for k in range(trial_number):
            sp_times = spike_times[i]['trials'][k]
            plt.plot(sp_times, np.ones(len(sp_times)) + k, 'k|')

        plt.xlim(0, stimulus_time[-1])
        plt.ylabel('Trial Number')
        plt.title('gap = ' + str(gap * 1000) + ' ms, pulse = ' + str(pulse_duration*1000) + ' ms, bin = ' + str(bin_width) +
                  ' ms')

        # PSTH
        plt.subplot(3, 1, 2)
        plt.plot(bin_edges[:-1], frate, 'k')
        plt.xlim(0, stimulus_time[-1])
        plt.ylabel('Firing Rate [Spikes/s]')

        # Stimulus
        plt.subplot(3, 1, 3)
        plt.plot(stimulus_time, stimulus_amplitude, 'k')
        plt.xlim(0, stimulus_time[-1])
        plt.yticks([0, 0, 80])
        plt.ylabel('Amplitude')
        plt.xlabel('time [ms]')

        # Save Plot to HDD
        figname = pathname + "RectIntervals_" + str(i) + '.png'
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Progress Bar
        percent = np.round(i / len(spike_times), 2)
        print('-- RectInterval Plots: %s %%  --' % (percent * 100))

    print('RectInterval Plots saved!')


# ----------------------------------------------------------------------------------------------------------------------
# VECTOR STRENGTH AND BOOTSTRAPPING


def bootstrapping_vs(path_names, nresamples, plot_histogram):
    # Load data
    data_name = path_names[0]
    pathname = path_names[2]
    fname = pathname + 'intervals_mas_spike_times.npy'
    fname2 = pathname + 'intervals_mas_vs.npy'
    spike_times = np.load(fname).item()
    vector_strength = np.load(fname2).item()
    resamples = {}
    # nresamples = 1000  # Number of resamples

    # Create Directory
    directory = os.path.dirname(pathname + os.path.join('vs', ''))
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    text_file_name = pathname + os.path.join('vs', 'bootstrap_vs.txt')

    try:
        text_file = open(text_file_name, 'r+')
        text_file.truncate(0)
    except FileNotFoundError:
        print('create new bootstrap_vs.txt file')

    for k in spike_times:
        if spike_times[k][0][1] < 0.0003:  # only tau = 0.1 ms
            gap = spike_times[k]['gap']
            data_noise = {}
            data = spike_times[k]['all_spike_times']
            vs_resamples = []
            # Now create resamples
            for n in range(nresamples):
                # This adds noise in the range of a to b [(b - a) * random() + a]
                a = -0.04
                b = 0.04
                noise = ((b-a) * np.random.random(len(data)) + a)
                d = data+noise

                # Vector Strength
                vs, phase = sg.vectorstrength(d, gap)
                vs_resamples = np.append(vs_resamples, vs)
                data_noise.update({n: {'spikes': d, 'vs': vs, 'phase': phase}})

            vs_mean = np.mean(vs_resamples)
            percentile_95 = np.percentile(vs_resamples, 95)

            # Histogram
            if plot_histogram:
                real_vs = vector_strength[k]['vs']
                y, binEdges = np.histogram(vs_resamples, bins=20)
                bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
                plt.hist(vs_resamples, bins=20, facecolor='grey')
                plt.plot(bincenters, y, 'k-')
                plt.plot([real_vs, real_vs], [0, np.max(y)], 'k--')
                plt.plot([percentile_95, percentile_95], [0, np.max(y)], 'r--')
                plt.xlabel('vector strength')
                plt.ylabel('count')
                plt.title('gap = %s ms' % str((gap*1000)))
                # Save Plot to HDD
                figname = pathname + "vs/vs_hist_" + str(gap*1000) + ".png"
                fig = plt.gcf()
                fig.set_size_inches(16, 12)
                fig.savefig(figname, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print('%s %% done' % str(((gap*1000)/15)*100))

            with open(text_file_name, 'a') as text_file:
                text_file.write('Gap: %s seconds\n' % gap)
                text_file.write('Boot. Mean: %s\n' % vs_mean)
                text_file.write('Boot. 95%% Percentile: %s\n' % percentile_95)
                text_file.write('Sample VS: %s\n' % vector_strength[k]['vs'])
                if vector_strength[k]['vs'] > percentile_95:
                    text_file.write('***\n')
                else:
                    text_file.write('n.s.\n')
                text_file.write('-----------------------------------------------\n')

            resamples.update({k: {'gap': gap, 'data': data_noise}})

    return 0


def bootstrap_test(datasets):
    # Load data
    data_name = datasets[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    fname = pathname + 'intervals_mas_spike_times.npy'
    fname2 = pathname + 'intervals_mas_vs.npy'
    spike_times = np.load(fname).item()
    vector_strength = np.load(fname2).item()
    resamples = {}
    nresamples = 1000  # Number of resamples
    for k in spike_times:
        if spike_times[k][0][1] < 0.0003:  # only tau = 0.1 ms
            gap = spike_times[k]['gap']
            data_noise = {}
            data = spike_times[k]['all_spike_times']
            vs_resamples = []
            # Now create resamples
            for n in range(nresamples):
                # This adds noise in the range of a to b [(b - a) * random() + a]
                noise_values = np.arange(0.001, 0.1, 0.001)
                percentile_95 = np.zeros((len(noise_values), 1))
                for aa in range(len(noise_values)):
                    a = noise_values[aa]
                    b = -a
                    noise = ((b-a) * np.random.random(len(data)) + a)
                    d = data+noise
                    vs, _ = sg.vectorstrength(d, gap)
                    percentile_95[aa] = np.percentile(vs, 95)

                plt.plot(noise_values, percentile_95)
                plt.show()
                embed()

    return 0


# ----------------------------------------------------------------------------------------------------------------------
# MISC

def getMetadataDict(bHandle):

    def unpackMetadata(sec):
        metadata = dict()
        metadata = {prop.name: sec[prop.name] for prop in sec.props}
        if hasattr(sec, 'sections') and len(sec.sections) > 0:
            metadata.update({subsec.name: unpackMetadata(subsec) for subsec in sec.sections})
        return metadata

    return unpackMetadata(bHandle.metadata)


def adjustSpines(ax, spines=['left', 'bottom'], shift_pos=False):
    for loc, spine in ax.spines.items():
        if loc in spines:
            if shift_pos:
                spine.set_position(('outward', 10))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def make_directory(dataset):

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/naturalmothcalls/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/batcalls/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/callseries/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/callseries/bats/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/callseries/moths/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory
    return 0


def tagtostimulus(path_names):
    dataset = path_names[0]
    pathname = path_names[1]
    tag_list = np.load(pathname + 'Calls_tag_list.npy')
    nix_file = path_names[3] + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    # mtags = {}
    connection = {}
    connection2 = {}
    for k in range(len(tag_list)):
        # mtags.update({tag_list[k]: b.multi_tags[tag_list[k]]})
        mtag = b.multi_tags[tag_list[k]]
        sound_name = mtag.metadata.sections[0][2]
        connection.update({sound_name: tag_list[k]})
        connection2.update({tag_list[k]: sound_name})

    f.close()
    return connection, connection2


def tagtostimulus_gap(path_names, protocol_name):
    dataset = path_names[0]
    pathname = path_names[1]
    tag_list = np.load(pathname + protocol_name + '_tag_list.npy')
    nix_file = path_names[3] + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    stim = {}
    stim_time = {}
    gap = {}
    pd = {}
    pr = {}
    fs = 1000
    for k in range(len(tag_list)):
        tag = b.tags[tag_list[k]]
        metadata = getMetadataDict(tag)
        for ii in metadata:
            if ii.startswith('data'):
                key = ii
        for kk in metadata[key]:
            if kk.startswith('dataset-settings'):
                key2 = kk
        period = metadata[key][key2]['period']
        pulse_duration = metadata[key][key2]['pulseduration']
        stimulus_duration = metadata[key][key2]['duration']
        total_amplitude = 1
        gap.update({k: str(np.ceil((period - pulse_duration) * 1000))})
        pd.update({k: str(pulse_duration*1000)})
        pr.update({k: str(period*1000)})
        t, s = rect_stimulus(period, pulse_duration, stimulus_duration, total_amplitude, fs, plotting=False)
        stim.update({k: s})
        stim_time.update({k: t})
    f.close()
    return stim, stim_time, gap, pd, pr


def pytomat(path_names, protocol_name):
    # Load Voltage Traces
    dataset = path_names[0]
    file_pathname = path_names[1]
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    # tag_list = np.load(tag_list_path)
    for i in range(1, 90):
        volt = voltage['SingleStimulus-file-'+str(i)][0]
        scipy.io.savemat('/media/brehm/Data/volt_test/volt' + str(i) + '.mat', {'volt': volt})
    return 0


def mattopy(stim_type, fs):

    if stim_type == 'all_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'moth_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_series_selected':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav']

    if stim_type == 'moth_single_selected':
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

    if stim_type == 'all_single':
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
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_single':
        stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_series':
        stims = ['callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    # file_pathname = '/media/brehm/Data/MasterMoth/stimuli/'
    file_pathname = '/media/nils/Data/Moth/stimuli/'
    # listing = os.listdir(file_pathname)
    samples = [[]] * len(stims)
    stim_names = [[]] * len(stims)
    for i in range(len(stims)):
        # file_name = file_pathname + stim_names[i][0:-4] + '/samples.mat'
        file_name = file_pathname + stims[i][0:-4] + '/samples.mat'
        mat_file = scipy.io.loadmat(file_name)
        samples[i] = sorted(np.append(mat_file['samples']['active'][0][0][0][:], mat_file['samples']['passive'][0][0][0][:]) / fs)
        stim_names[i] = stims[i][0:-4]
    return samples, stim_names


def get_session_metadata(datasets):
    # Copies info.dat to the analysis folder
    for dat in range(len(datasets)):
        try:
            data_name = datasets[dat]
            pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
            info_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + 'info.dat'

            copyfile(info_file, pathname+'info.dat')
            print('Copied info.dat of %s' % data_name)
        except FileNotFoundError:
            print('File Not Found: %s' % data_name)

    return 0


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

# ----------------------------------------------------------------------------------------------------------------------
# GAP PARADIGM


def gap_analysis(dataset, protocol_name):
    # No longer in use!
    # Load Voltage Traces
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    tag_list = np.load(tag_list_path)
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Set time range of single trials
    cut = np.arange(0.6, 17, 1.5)
    cut_samples = cut * fs
    voltage_new = {}
    for k in range(len(tag_list)):
        v = {}
        volt = voltage[tag_list[k]]
        # Now cut out the single trials
        for i in range(len(cut_samples)-1):
            v.update({i: volt[int(cut_samples[i]):int(cut_samples[i+1])]})
        voltage_new.update({tag_list[k]: v})
    # Save Voltage to HDD (This overwrites the input voltage data)
    dname = file_pathname + protocol_name + '_voltage.npy'
    np.save(dname, voltage_new)
    print('Saved voltage (Overwrite Input Voltage)')

    return voltage_new


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Junkyard

def plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data # ', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

    return 0


def detect_peaks(x, peak_params):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    peak_params: All the parameters listed below in a dict.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
        if mph = 'dynamic': this value will be computed automatically
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    maxph: maximal peak height: Peaks > maxph are removed. maxph = 'dynamic': maxph is computed automatically
    filter_on: Bandpass filter

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    modified by Nils Brehm 2018
    """

    # Set Parameters
    mph = peak_params['mph']
    mpd = peak_params['mpd']
    valley = peak_params['valley']
    show = peak_params['show']
    maxph = peak_params['maxph']
    filter_on = peak_params['filter_on']
    threshold = 0
    edge = 'rising'
    kpsh = False
    ax = None


    # Filter Voltage Trace
    if filter_on:
        fs = 100 * 1000
        nyqst = 0.5 * fs
        lowcut = 300
        highcut = 2000
        low = lowcut / nyqst
        high = highcut / nyqst
        x = voltage_trace_filter(x, [low, high], ftype='band', order=4, filter_on=True)

    # Dynamic mph
    if mph == 'dynamic':
        mph = 2 * np.median(abs(x)/0.6745)

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    # remove peaks > maximum peak height
    if ind.size and maxph is not None:
        if maxph == 'dynamic':
            hist, bins = np.histogram(x[ind])
            # maxph = np.round(np.max(x)-50)
            # idx = np.where(hist > 10)[0][-1]
            maxph = np.round(bins[-2]-20)
        ind = ind[x[ind] <= maxph]

    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind


def peak_seek(x, mpd, mph):
    # Find all maxima and ties
    localmax = (np.diff(np.sign(np.diff(x))) < 0).nonzero()[0] + 1
    locs = localmax[x[localmax] > mph]

    while 1:
        idx = np.where(np.diff(locs) < mpd)
        idx = idx[0]
        if not idx.any():
            break
        rmv = list()
        for i in range(len(idx)):
            a = x[locs[idx[i]]]
            b = x[locs[idx[i]+1]]
            if a > b:
                rmv.append(True)
            else:
                rmv.append(False)
        locs = locs[idx[rmv]]

    embed()
    #locs = find(x(2:end - 1) >= x(1: end - 2) & x(2: end - 1) >= x(3: end))+1;


def indexes(y, dynamic, th_factor=2, min_dist=50, maxph=0.8, th_window=200):
    """Peak detection routine.
    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.
    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    th_factor : trheshold = th_factor * median(y / 0.6745)
        Only the peaks with amplitude higher than the threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    maxph: All peaks larger than maxph * max(y) are removed.
    th_window: end point of the range for computing threshold values

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(th_window, str):
        th_window = -1
    # thres = th_factor * np.median(abs(y[0:th_window]) / 0.6745)
    if dynamic:
        thres = th_factor * np.median(abs(y[0:th_window] - np.median(y[0:th_window])))
    else:
        thres = th_factor
    maxph = np.max(abs(y)) * maxph

    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    # thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros] = zerosr[zeros]
        zeros, = np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros] = zerosl[zeros]
        zeros, = np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres)
                     & (y < maxph))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks, thres


def get_spike_times(dataset, protocol_name, peak_params, show_detection):
    """Get Spike Times using the detect_peak() function.

    Notes
    ----------
    This function gets all the spike times in the voltage traces loaded from HDD.

    Parameters
    ----------
    dataset :       Data set name (string)
    protocol_name:  protocol name (string)
    peak_params:    Parameters for spike detection (dict)
    show_detection: If true spike detection of first trial is shown (boolean)

    Returns
    -------
    spikes: Saves spike times (in seconds) to HDD in a .npy file (dict).

    """

    # Load Voltage Traces
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    tag_list = np.load(tag_list_path)
    spikes = {}
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Loop trough all tags in tag_list
    for i in range(len(tag_list)):
        trials = len(voltage[tag_list[i]])
        spike_times = [list()] * trials
        for k in range(trials):  # loop trough all trials
            if k == 0 & show_detection is True:  # For all tags plot the first trial spike detection
                peak_params['show'] = True
                spike_times[k] = detect_peaks(voltage[tag_list[i]][k], peak_params) / fs  # in seconds
            else:
                peak_params['show'] = False
                spike_times[k] = detect_peaks(voltage[tag_list[i]][k], peak_params) / fs  # in seconds
        spikes.update({tag_list[i]: spike_times})
    # Save to HDD
    file_name = file_pathname + protocol_name + '_spikes.npy'
    np.save(file_name, spikes)
    print('Spike Times saved (protocol: ' + protocol_name + ')')

    return 0


def spike_times_indexes(dataset, protocol_name, th_factor, min_dist, maxph, show, save_data):
    """Get Spike Times using the indexes() function.

    Notes
    ----------
    This function gets all the spike times in the voltage traces loaded from HDD.

    Parameters
    ----------
    dataset :       Data set name (string)
    protocol_name:  protocol name (string)
    th_factor: threshold = th_factor * median(abs(x)/0.6745)
    min_dist: Min. allowed distance between two spikes
    maxph: Peaks larger than maxph * max(x) are removed

    Returns
    -------
    spikes: Saves spike times (in seconds) to HDD in a .npy file (dict).

    """

    # Load Voltage Traces
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    tag_list = np.load(tag_list_path)
    spikes = {}
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Loop trough all tags in tag_list
    for i in range(len(tag_list)):
        trials = len(voltage[tag_list[i]])
        spike_times = [list()] * trials
        for k in range(trials):  # loop trough all trials
            spike_times[k] = indexes(voltage[tag_list[i]][k], th_factor=th_factor, min_dist=min_dist, maxph=maxph)
            if k == 0 and show:
                plt.plot(voltage[tag_list[i]][k], 'k')
                plt.plot(spike_times[k], voltage[tag_list[i]][k][spike_times[k]], 'ro')
                plt.show()
            spike_times[k] = spike_times[k] / fs  # in seconds
        spikes.update({tag_list[i]: spike_times})

    # Save to HDD
    if save_data:
        file_name = file_pathname + protocol_name + '_spikes.npy'
        np.save(file_name, spikes)
        print('Spike Times saved (protocol: ' + protocol_name + ')')

    return spikes


def square_wave(period, pulse_duration, stimulus_duration, sampling_rate):
    # -- Square Wave --------------------------------------------------------------------------------------------------
    # Unit: time in msec, frequency in Hz, dutycycle in %, value = 0: disabled
    # N = 1000  # sample count
    # freq = 0
    # dutycycle = 0
    # period = 200
    # pulseduration = 50
    sampling_rate = 800*1000
    n = sampling_rate
    t = np.linspace(0, stimulus_duration, stimulus_duration * sampling_rate, endpoint=False)
    sw = np.arange(n) % period < pulse_duration  # Returns bool array (True=1, False=0)
    return t, sw


def view_nix(dataset):
    # Open Nix File
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + dataset + '/' + dataset + '.nix'
    try:
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        print('".nix" extension found')
        print(dataset)
    except OSError:
        print('File is damaged!')
        return 1
    except RuntimeError:
        try:
            f = nix.File.open(nix_file + '.h5', nix.FileMode.ReadOnly)
            print('".nix.h5" extension found')
            print(dataset)
        except RuntimeError:
            print(dataset)
            print('File not found')
            return 1

    b = f.blocks[0]

    # Create Text File:
    text_file_name = file_pathname + 'stimulus.txt'

    try:
        text_file = open(text_file_name, 'r+')
        text_file.truncate(0)
    except FileNotFoundError:
        print('create new stimulus.txt file')

    # Get all multi tags
    mtags = b.multi_tags

    # print('\nMulti Tags found:')
    with open(text_file_name, 'a') as text_file:
        text_file.write(dataset + '\n\n')
        text_file.write('Multi Tags found: \n')
        for i in range(len(mtags)):
            # print(mtags[i].name)
            text_file.write(mtags[i].name + '\n')

    # Get all tags
    tags = b.tags
    with open(text_file_name, 'a') as text_file:
        # print('\nTags found:')
        text_file.write('\nTags found: \n')
        for i in range(len(tags)):
            # print(tags[i].name)
            text_file.write(tags[i].name + '\n')

    f.close()
    return 0


def get_spiketimes_from_nix(tag,mtag,dataset):
    if mtag == 1:
        spike_times = tag.retrieve_data(dataset, 1)  # from multi tag get spiketimes (dataset,datatype)
    else:
        spike_times = tag.retrieve_data(1)  # from single tag get spiketimes
    return spike_times


def get_voltage(tag, mtag, dataset):
    if mtag == 1:

        volt = tag.retrieve_data(dataset, 0)  # for multi tags
    else:
        volt = tag.retrieve_data(0)  # for single tags
    return volt


def inst_firing_rate(spikes: object, tmax: object, dt: object) -> object:
    time = np.arange(0, tmax, dt)
    rate = np.zeros((time.shape[0]))
    isi = np.diff(np.insert(spikes, 0, 0))
    inst_rate = 1 / isi
    spikes_id = np.round(spikes / dt)
    spikes_id = np.insert(spikes_id, 0, 0)
    spikes_id = spikes_id.astype(int)

    for i in range(spikes_id.shape[0] - 1):
        rate[spikes_id[i]:spikes_id[i + 1]] = inst_rate[i]
    return time, rate


def loading_fifield_data(pathname):
    # Load data: FIField and FICurves
    file_name = pathname + 'FICurve.npy'
    ficurve = np.load(file_name).item()  # Load dictionary data

    file_name2 = pathname + 'FIField.npy'
    fifield = np.load(file_name2)  # Load data array

    file_name3 = pathname + 'frequencies.npy'
    frequencies = np.load(file_name3)

    return ficurve, fifield, frequencies


def fifield_analysis(datasets, spike_threshold, peak_params):
    for all_datasets in range(len(datasets)):
        nix_name = datasets[all_datasets]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + nix_name + "/"
        filename = pathname + 'FIField_voltage.npy'

        # Load data from HDD
        fifield_volt = np.load(filename).item()  # Load dictionary data
        freqs = np.load(pathname + 'frequencies.npy')
        amps = np.load(pathname + 'amplitudes.npy')
        amps_uni = np.unique(np.round(amps))
        freqs_uni = np.unique(freqs) / 1000

        # Find Spikes
        fi = {}
        for f in range(len(freqs_uni)):  # Loop through all different frequencies
            dbSPL = {}
            for a in fifield_volt[freqs_uni[f]].keys():  # Only loop through amps that exist
                repeats = len(fifield_volt[freqs_uni[f]][a]) - 1
                spike_count = np.zeros(repeats)
                for trial in range(repeats):
                    x = fifield_volt[freqs_uni[f]][a][trial]
                    spike_times = indexes(x, th_factor=2, min_dist=50, maxph=0.8)
                    '''
                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'],
                                               show=peak_params['show'], ax=None, maxph=peak_params['maxph'],
                                               dynamic=peak_params['dynamic'], filter_on=peak_params['filter_on'])
                    '''
                    spike_count[trial] = len(spike_times)
                # m = np.mean(spike_count)
                # std = np.std(spike_count)
                dummy = [np.mean(spike_count), np.std(spike_count), repeats]
                dbSPL.update({a: dummy})
            fi.update({freqs_uni[f]: dbSPL})

        # Collect data for FI Curves and FIField
        dbSPL_threshold = np.zeros((len(freqs_uni), 3))
        for f in range(len(freqs_uni)):
            amplitude_sorted = sorted(fi[freqs_uni[f]])
            mean_spike_count = np.zeros((len(amplitude_sorted)))
            std_spike_count = np.zeros((len(amplitude_sorted)))
            k = 0
            for i in amplitude_sorted:
                mean_spike_count[k] = fi[freqs_uni[f]][i][0]
                std_spike_count[k] = fi[freqs_uni[f]][i][1]
                k += 1

            # Find db SPL threshold
            th = mean_spike_count >= spike_threshold
            dbSPL_threshold[f, 0] = freqs_uni[f]
            if th.any():
                dbSPL_threshold[f, 1] = amplitude_sorted[np.min(np.where(th))]
            else:
                dbSPL_threshold[f, 1] = '100'  # no threshold could be found

            # Save FIField data to HDD
            dname1 = pathname + 'FICurve_' + str(freqs_uni[f])
            np.savez(dname1, amplitude_sorted=amplitude_sorted, mean_spike_count=mean_spike_count, std_spike_count=std_spike_count,
                     spike_threshold=spike_threshold, dbSPL_threshold=dbSPL_threshold, freq=freqs_uni[f])
            # Plot FICurve for this frequency
            # plot_ficurve(amplitude_sorted, mean_spike_count, std_spike_count, freqs_uni[f], spike_threshold,
            #                pathname, savefig=True)

            # Estimate Progress
            percent = np.round(f / len(freqs_uni), 2)
            print('--- %s %% done ---' % (percent * 100))

        # Save FIField data to HDD
        dname = pathname + 'FIField_plotdata.npy'
        np.save(dname, dbSPL_threshold)

        dname2 = pathname + 'FICurves.npy'
        np.save(dname2, fi)

        # Plot FIField and save it to HDD
        # plot_fifield(dbSPL_threshold, pathname, savefig=True)

        print('Analysis finished: %s' % datasets[all_datasets])

        # Progress Bar
        percent = np.round((all_datasets + 1) / len(datasets), 2)
        print('-- Analysis total: %s %%  --' % (percent * 100))
    return 0


def compute_fifield(data_name, data_file, tag, spikethreshold):
    # Old version
    # Open the nix file
    start_time = time.time()
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]

    # Get tags
    mtag = b.multi_tags[tag]

    # Get Meta Data
    meta = mtag.metadata.sections[0]
    duration = meta["Duration"]
    frequency = mtag.features[5].data[:]
    amplitude = mtag.features[4].data[:]
    final_data = np.zeros((len(frequency), 3))

    # Create Directory for Saving Data
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    # Get data for all frequencies
    freq = np.unique(frequency)  # All used frequencies in experiment
    qq = 0
    ficurve = {}
    db_threshold = np.zeros((len(freq), 2))
    timeneeded = np.zeros((len(freq)))
    rawdata = {}
    for q in freq:
        intertime1 = time.time()
        d = np.where(frequency == q)[0][:]
        k = 0
        data = np.zeros((len(d), 4))
        for j in d:
            spikes = mtag.retrieve_data(int(j), 1)[:]  # Spike Times
            isi = np.diff(spikes)  # Inter Spike Intervals
            spikecount = len(mtag.retrieve_data(int(j), 1)[:])  # Number of spikes per stimulus presentation
            data[k, 0] = frequency[j]
            data[k, 1] = amplitude[j]
            data[k, 2] = spikecount
            if isi.any():
                data[k, 3] = np.mean(isi)
            else:
                data[k, 3] = 0
            k += 1

        # Now average all spike counts per amplitude
        amps = np.unique(data[:, 1])  # All used amplitudes in experiment
        k = 0
        average = np.zeros((len(amps), 4))
        for i in amps:
            ids = np.where(data == i)[0]
            average[k, 0] = i  # Amplitude
            average[k, 1] = np.mean(data[ids, 2])  # Mean spike number per presentation
            average[k, 2] = np.std(data[ids, 2])  # STD spike number per presentation
            average[k, 3] = np.mean(data[ids, 3])  # Mean mean inter spike interval
            k += 1

        # Now find dB SPL threshold for FIField
        db_threshold[qq, 0] = q
        dummyID = np.where(average[:, 1] >= spikethreshold)
        if dummyID[0].any():
            db_threshold[qq, 1] = average[np.min(dummyID), 0]
        else:
            db_threshold[qq, 1] = 100

        # Now store FICurve data in dictionary
        ficurve.update({int(q): average})
        rawdata.update({int(q): data})

        # Estimate Time Remaining for Analysis
        percent = np.round((qq + 1) / len(freq), 3)
        print('--- %s %% done ---' % (percent * 100))
        intertime2 = time.time()
        timeneeded[qq] = np.round((intertime2 - intertime1) / 60, 2)
        if qq > 0:
            timeremaining = np.round(np.mean(timeneeded[0:qq]), 2) * (len(freq) - qq)
        else:
            timeremaining = np.round(np.mean(timeneeded[qq]), 2) * (len(freq) - qq)
        print('--- Time remaining: %s minutes ---' % timeremaining)
        qq += 1

    # Save Data to HDD
    dname = pathname + 'FICurve.npy'
    dname2 = pathname + 'FIField.npy'
    dname3 = pathname + 'frequencies.npy'
    dname4 = pathname + 'rawdata.npy'
    np.save(dname4, rawdata)
    np.save(dname3, freq)
    np.save(dname2, db_threshold)
    np.save(dname, ficurve)

    f.close()  # Close Nix File
    print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60))
    return 'FIField successfully computed'


def fifield_voltage(data_name, data_file, tag):
    # Open the nix file
    start_time = time.time()
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]

    # Get tags
    mtag = b.multi_tags[tag]

    # Get Meta Data
    # meta = mtag.metadata.sections[0]
    # duration = meta["Duration"]
    frequency = mtag.features[5].data[:]
    amplitude = mtag.features[4].data[:]

    # Create Directory for Saving Data
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    # Get data for all frequencies
    freq = np.unique(frequency)  # All used frequencies in experiment
    # amps = np.unique(amplitude)
    qq = 0
    fifield_volt = {}
    for q in freq:
        volt2 = {}
        ids_freq = np.where(frequency == q)[0][:]
        amps = np.unique(amplitude[ids_freq])  # Used db SPL with this frequency
        for a in amps:
            volt1 = {}
            id_amp = np.where(amplitude == a)[0][:]
            for i in range(len(id_amp)):
                v = mtag.retrieve_data(int(id_amp[i]), 0)[:]  # Voltage Trace
                volt1.update({i: v})
            volt1.update({i+1: a})
            volt2.update({int(np.round(a)): volt1})
        fifield_volt.update({int(q/1000): volt2})

        # Estimate Progress
        percent = np.round((qq + 1) / len(freq), 3)
        print('--- %s %% done ---' % (percent * 100))
        qq += 1

    # Save Data to HDD
    dname = pathname + 'FIField_voltage.npy'
    np.save(dname, fifield_volt)
    dname2 = pathname + 'frequencies.npy'
    np.save(dname2, frequency)
    dname3 = pathname + 'amplitudes.npy'
    np.save(dname3, amplitude)

    f.close()  # Close Nix File
    return fifield_volt, frequency, amplitude


def get_fifield_data(datasets):
    for all_datasets in range(len(datasets)):
        data_name = datasets[all_datasets]
        nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + data_name + '.nix'
        tag = 'FIField-sine_wave-1'

        # Read Voltage Traces from nix file and save it to HDD
        volt, freq, amp = fifield_voltage(data_name, nix_file, tag)
        print('Got Data set: %s and saved it' % datasets[all_datasets])
        percent = np.round((all_datasets + 1) / len(datasets), 2)
        print('-- Getting data total: %s %%  --' % (percent * 100))

    return volt, freq, amp


def soundfilestimuli_spike_detection(datasets, peak_params):
    data_name = datasets[0]
    # stim_sets = [['/mothsongs/'], ['/batcalls/noisereduced/']]
    stim_sets = '/naturalmothcalls/'
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + stim_sets
    file_list = os.listdir(pathname)
    dt = 100 * 1000
    # Load voltage data
    for k in range(len(file_list)):
        if file_list[k][-11:] == 'voltage.npy':
            # Load Voltage from HDD
            file_name = pathname + file_list[k]
            voltage = np.load(file_name).item()
            sp = {}

            # Go through all trials and detect spikes
            trials = len(voltage)

            for i in range(trials):
                x = voltage[i]
                spike_times = detect_peaks(x, peak_params)
                spike_times = spike_times / dt  # Now spikes are in seconds
                sp.update({i: spike_times})

                # Save Spike Times to HDD
                dname = file_name[:-12] + '_spike_times.npy'
                np.save(dname, sp)

    print('Spike Times saved')
    return 0


def soundfilestimuli_spike_distance(datasets):
    data_name = datasets[0]
    for p in ['/mothsongs/', '/batcalls/noisereduced/']:
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + p
        file_list = os.listdir(pathname)

        # Load spike times
        for k in range(len(file_list)):
            if file_list[k][-15:] == 'spike_times.npy':
                file_name = pathname + file_list[k]
                spike_times = np.load(file_name).item()
                duration = spike_times[0][-1] + 0.01
                dt = 100 * 1000
                tau = 0.005
                print('Stim: %s' % file_name[29:])
                d = spike_train_distance(spike_times[0], spike_times[1], dt, duration, tau, plot=False)
                print('\n')


def vanrossum_matrix_backup(dataset, tau, duration, dt_factor, boot_sample, stim_type, save_fig):
    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    spikes = np.load(pathname + 'Calls_spikes.npy').item()
    # tag_list = np.load(pathname + 'Calls_tag_list.npy')
    '''
    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
             'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
             'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']


    stims = ['naturalmothcalls/BCI1062_07x07.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav',
             'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/neritos_cotes_07x07.wav']


    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav',
             'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav']


    stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                'batcalls/Eptesicus_nilssonii_1_s.wav',
                'batcalls/Myotis_bechsteinii_1_n.wav',
                'batcalls/Myotis_brandtii_1_n.wav',
                'batcalls/Myotis_nattereri_1_n.wav',
                'batcalls/Nyctalus_leisleri_1_n.wav',
                'batcalls/Nyctalus_noctula_2_s.wav',
                'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                'batcalls/Vespertilio_murinus_1_s.wav']

    stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
             'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
             'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
             'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
             'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
             'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
             'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
             'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
             'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
             'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav',
             'batcalls/Barbastella_barbastellus_1_n.wav',
             'batcalls/Eptesicus_nilssonii_1_s.wav',
             'batcalls/Myotis_bechsteinii_1_n.wav',
             'batcalls/Myotis_brandtii_1_n.wav',
             'batcalls/Myotis_nattereri_1_n.wav',
             'batcalls/Nyctalus_leisleri_1_n.wav',
             'batcalls/Nyctalus_noctula_2_s.wav',
             'batcalls/Pipistrellus_pipistrellus_1_n.wav',
             'batcalls/Pipistrellus_pygmaeus_2_n.wav',
             'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
             'batcalls/Vespertilio_murinus_1_s.wav']
    '''

    if stim_type == 'selected':
        stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']

    if stim_type == 'series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']
    # Tags and Stimulus names

    connection = tagtostimulus(dataset)
    stimulus_tags = [''] * len(stims)
    for p in range(len(stims)):
        stimulus_tags[p] = connection[stims[p]]

    # Convert all Spike Trains to e-pulses
    # trial_nr = int(len(spikes[stimulus_tags[0]]))
    # trial_nr = 20
    trains = {}

    for k in range(len(stimulus_tags)):
        trial_nr = len(spikes[stimulus_tags[k]])
        tr = [[]] * trial_nr
        for j in range(trial_nr):
            x = spikes[stimulus_tags[k]][j]
            # tr.update({j: spike_e_pulses(x, dt_factor, tau)})
            tr[j] = spike_e_pulses(x, dt_factor, tau, duration)
        trains.update({stimulus_tags[k]: tr})

        # for i in range(len(stims)):
        # print(str(i) + ': ' + stims[i])

    call_count = len(stimulus_tags)
    # Select Template and Probes and bootstrap this process
    mm = {}
    for boot in range(boot_sample):
        count = 0
        match_matrix = np.zeros((call_count, call_count))
        templates = {}
        probes = {}
        # rand_ids = np.random.randint(trial_nr, size=call_count)
        for i in range(call_count):
            trial_nr = len(spikes[stimulus_tags[i]])
            rand_id = np.random.randint(trial_nr, size=1)
            idx = np.arange(0, trial_nr, 1)
            # idx = np.delete(idx, rand_ids[i])
            # templates.update({i: trains[stimulus_tags[i]][rand_ids[i]]})
            idx = np.delete(idx, rand_id[0])
            templates.update({i: trains[stimulus_tags[i]][rand_id[0]]})
            for q in range(len(idx)):
                probes.update({count: [trains[stimulus_tags[i]][idx[q]], i]})
                count += 1

        # Compute VanRossum Distance
        for pr in range(len(probes)):
            d = np.zeros(len(templates))
            for tmp in range(len(templates)):
                d[tmp] = vanrossum_distance(templates[tmp], probes[pr][0], dt_factor, tau)

            # What happens if there are two mins? - The first one is taken

            template_match = np.where(d == np.min(d))[0][0]
            song_id = probes[pr][1]
            match_matrix[template_match, song_id] += 1
            '''
            if len(np.where(d == np.min(d))) > 1:
                print('Found more than 1 min.')
            if pr == 0:
                print(d)
                print('---------------')

                plt.figure()
                plt.subplot(3, 1, 1)
                plt.plot(templates[template_match])
                plt.title('Matched Template: ' + str(d[template_match]))
                plt.subplot(3, 1, 2)
                plt.plot(templates[song_id])
                plt.title('Probes correct Template: ' + str(d[song_id]))
                plt.subplot(3, 1, 3)
                plt.plot(probes[pr][0])
                plt.title('Probe')
                plt.show()
                embed()
            '''
        mm.update({boot: match_matrix})

    mm_mean = sum(mm.values()) / len(mm)

    # Percent Correct
    percent_correct = np.zeros((len(mm_mean)))
    correct_nr = np.zeros((len(mm_mean)))
    for r in range(len(mm_mean)):
        percent_correct[r] = mm_mean[r, r] / np.sum(mm_mean[:, r])
        correct_nr[r] = mm_mean[r, r]

    correct_matches = np.sum(correct_nr) / np.sum(mm_mean)

    if save_fig:
        # Plot Matrix
        plt.imshow(mm_mean)
        plt.xlabel('Original Calls')
        plt.ylabel('Matched Calls')
        plt.colorbar()
        plt.xticks(np.arange(0, len(mm_mean), 1))
        plt.yticks(np.arange(0, len(mm_mean), 1))
        plt.title('tau = ' + str(tau * 1000) + ' ms' + ' T = ' + str(duration * 1000) + ' ms')

        # Save Plot to HDD
        figname = "/media/brehm/Data/MasterMoth/figs/" + dataset + '/VanRossumMatrix_' + str(tau * 1000) + \
                  '_' + str(duration * 1000) + '.png'
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('tau = ' + str(tau * 1000) + ' ms' + ' T = ' + str(duration * 1000) + ' ms done')

    return mm_mean, correct_matches


def moth_intervals_spike_detection_backup(path_names, window=None, th_factor=1, mph_percent=0.8, filter_on=True, save_data=True, show=True):
    # Load data
    fs = 100 * 1000
    data_name = path_names[0]
    pathname = path_names[1]
    fname = pathname + 'intervals_mas_voltage.npy'
    voltage = np.load(fname).item()

    # Now detect spikes in each trial and update input data
    for i in voltage:
        spikes = []
        stimulus = voltage[i]['stimulus']
        stimulus_time = voltage[i]['stimulus_time']
        info = str(voltage[i]['gap']*1000)
        trial_number = voltage[i]['trials']
        spike_times = [[]] * trial_number
        spike_times_valley = [[]] * trial_number

        for k in range(trial_number):
            x = voltage[i][k][0]
            # Filter Voltage Trace
            if filter_on:
                nyqst = 0.5 * fs
                lowcut = 300
                highcut = 2000
                low = lowcut / nyqst
                high = highcut / nyqst
                x = voltage_trace_filter(x, [low, high], ftype='band', order=2, filter_on=True)

            th = pk.std_threshold(x, fs, window, th_factor)
            spike_times[k], spike_times_valley[k] = pk.detect_peaks(x, th)

            # Remove large spikes
            t = np.arange(0, len(x) / fs, 1 / fs)
            spike_size = pk.peak_size_width(t, x, spike_times[k], spike_times_valley[k], pfac=0.75)
            spike_times[k], spike_times_valley[k], marked, marked_valley = \
                remove_large_spikes(x, spike_times[k], spike_times_valley[k], mph_percent=mph_percent, method='std')

            # Plot Spike Detection
            # Cut out spikes
            fs = 100 * 1000
            snippets = pk.snippets(x, spike_times[k], start=-100, stop=100)
            snippets_removed = pk.snippets(x, marked, start=-100, stop=100)

            if show and k == 0:
                plot_spike_detection_gaps(x, spike_times[k], spike_times_valley[k], marked, spike_size, mph_percent,
                                          snippets, snippets_removed, th, window, info, stimulus_time, stimulus)

            spike_times[k] = spike_times[k] / fs  # Now spike times are in real time (seconds)
            spike_times_valley[k] = spike_times_valley[k] / fs
            spikes = np.append(spikes, spike_times[k])  # Put spike times of each trial in one long array
            spike_count = len(spike_times[k])
            voltage[i][k].update({'spike_times': spike_times[k], 'spike_count': spike_count})
        voltage[i].update({'all_spike_times': spikes})

        # Get mean firing rate over all trials
        # bin_size = float(gap[i])/1000
        bin_size = 0.005
        n = len(spike_times)
        f_rate, b = psth(spike_times, n, bin_size, plot=False, return_values=True, separate_trials=True)
        mean_rate = np.mean(f_rate)

        # Possion Spikes
        nsamples = 100
        tmax = 0.5
        p_spikes = poission_spikes(nsamples, mean_rate, tmax)

        # Plot VS
        gap = voltage[i]['gap']*1000
        VS_plot = True
        if VS_plot:
            pp = np.arange(0.001, 0.1, 0.001)
            vs = np.zeros(shape=(len(spike_times), len(pp)))
            phase = np.zeros(shape=(len(spike_times), len(pp)))
            for q in range(len(spike_times)):
                for p in range(len(pp)):
                    vs[q, p], phase[q, p] = sg.vectorstrength(spike_times[q], pp[p])

            vs_mean = vs.mean(axis=0)
            th = pk.std_threshold(vs_mean, th_factor=1)
            peaks, _ = pk.detect_peaks(vs_mean, th)

            # Poisson (bootstrap) VS
            vs_boot = np.zeros(shape=(len(p_spikes), len(pp)))
            phase_boot = np.zeros(shape=(len(p_spikes), len(pp)))
            for j in range(len(p_spikes)):
                for p in range(len(pp)):
                    vs_boot[j, p], phase_boot[j, p] = sg.vectorstrength(p_spikes[j], pp[p])
            vs_boot_mean = vs_boot.mean(axis=0)
            vs_95 = np.percentile(vs_boot, 95, axis=0)

            plt.plot(pp * 1000, vs_mean, 'k')
            plt.plot(pp * 1000, vs_boot_mean, 'g')
            plt.plot(pp * 1000, vs_95, 'g--')
            plt.plot(pp[peaks] * 1000, vs_mean[peaks], 'ro')
            plt.plot([gap, gap], [0, np.max(vs_mean)], 'bx--')
            plt.xlabel('Gap [ms]')
            plt.ylabel('Mean VS')
            plt.title('gap =' + str(gap) + ', f_rate = ' + str(np.round(mean_rate)) + ' Hz')
            plt.show()
    # Save detected Spikes to HDD
    if save_data:
        dname = pathname + 'intervals_mas_spike_times.npy'
        np.save(dname, voltage)

    print('Spike Detection done')
    return 0


def vanrossum_distance2(probes, temp_idx, dt_factor, tau, duration):

    e_pulse_fs = tau / dt_factor
    duration_in_samples = np.array(duration) / e_pulse_fs
    duration_in_samples = duration_in_samples.astype(int)
    difference = [[]] * len(probes)
    for k in range(len(probes)):  # k is stimulus tag (a.k.a call)
        # Find the template
        template = probes[k][np.int(temp_idx[k])]
        diff2 = [[]] * len(probes)
        for i in range(len(probes)):  # loop through all trials of all calls
            # Select one probe
            diff = [[]] * len(probes[i])
            for j in range(len(probes[i])):
                g = probes[i][j]
                # Make sure both trains have same length
                difference_in_length = abs(len(template) - len(g))
                if len(template) > len(g):
                    g = np.append(g, np.zeros(difference_in_length))
                elif len(template) < len(g):
                    template = np.append(template, np.zeros(difference_in_length))
                # Compute difference
                diff[j] = template - g
            diff2[i] = diff
        difference[k] = diff2

    # Compute VanRossum Distance for each duration
    dt = tau/dt_factor
    distance_duration = {}
    for k in range(len(duration_in_samples)):
        distance = [[]] * len(difference)
        for j in range(len(difference)):
            d2 = [[]] * len(difference[j])
            for p in range(len(difference[j])):
                d = [[]] * len(difference[j][p])
                for i in range(len(difference[j][p])):
                    d[i] = np.sum((difference[j][p][i][0:duration_in_samples[k]]) ** 2) * (dt / tau)
                    if d[i] == 0:
                        d[i] = np.nan
                d2[p] = d
            distance[j] = d2
        distance_duration.update({duration[k]: distance})

    # f_count = np.sum(f)* (dt / tau)
    # g_count = np.sum(g) * (dt / tau)
    return distance_duration


def spike_distance_matrix(datasets):
    data_name = datasets[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/mothsongs/'
    stim1_name = 'Creatonotos03_series'
    stim2_name = 'Carales_series'

    file_name1 = stim1_name + '_spike_times.npy'
    file_name2 = stim2_name + '_spike_times.npy'

    stim1_dur = np.load(pathname + stim1_name + '_time.npy').item()
    stim2_dur = np.load(pathname + stim2_name + '_time.npy').item()

    file1 = np.load(pathname + file_name1).item()
    file2 = np.load(pathname + file_name2).item()

    duration = file1[0][-1] + 0.01
    dt = 100 * 1000
    # tau = 0.005
    tau = float(input('tau in ms: ')) / 1000
    d1 = np.zeros((len(file1), len(file1)))
    d2 = np.zeros((len(file2), len(file2)))
    dd = np.zeros((len(file1), len(file2)))

    # Stim 1 vs Stim 1
    for i in range(len(file1)):
        for j in range(len(file1)):
            d1[i, j] = spike_train_distance(file1[i], file1[j], dt, duration, tau, plot=False)
    d1[d1 == 0] = 'nan'

    # Stim 2 vs Stim 2
    for i in range(len(file2)):
        for j in range(len(file2)):
            d2[i, j] = spike_train_distance(file2[i], file2[j], dt, duration, tau, plot=False)
    d2[d2 == 0] = 'nan'

    # Stim 1 vs Stim 2
    for i in range(len(file1)):
        for j in range(len(file2)):
            dd[i, j] = spike_train_distance(file1[i], file2[j], dt, duration, tau, plot=False)

    # Look at trains of stim 1
    a1 = np.zeros((len(file1), 4))
    for k in range(len(file1)):
        a1[k, :2] = [np.nanmean(d1[k, :]), np.mean(dd[k, :])]
        if a1[k, 0] < a1[k, 1]:
            a1[k, 2] = 1
        else:
            a1[k, 2] = 2
        a1[k, 3] = abs(a1[k, 0] - a1[k, 1])

    # Look at trains of stim 2
    a2 = np.zeros((len(file2), 4))
    for k in range(len(file2)):
        a2[k, :2] = [np.nanmean(d2[:, k]), np.mean(dd[:, k])]
        if a2[k, 0] < a2[k, 1]:
            a2[k, 2] = 2
        else:
            a2[k, 2] = 1
        a2[k, 3] = abs(a2[k, 0] - a2[k, 1])

    print('spike trains %s (stim 1):' % stim1_name)
    print('cols: Mean Distance stim 1 | Mean Distance stim 2 | Classifed to | Difference')
    print(a1)
    print('\n')
    print('spike trains %s (stim 2):' % stim2_name)
    print('cols: Mean Distance stim 1 | Mean Distance stim 2 | Classifed to | Difference')
    print(a2)
    print('\n')
    print('Stim 1 duration: %s ms' % (len(stim1_dur[0])/(100*1000)))
    print('Stim 2 duration: %s ms' % (len(stim2_dur[0])/(100*1000)))

    return 0


def spike_e_pulses(spike_times, dt_factor, tau, duration, whole_train, method):
    # tau in seconds
    # dt_factor: dt = tau/dt_factor
    # spike times in seconds
    # duration in seconds
    old_settings = np.seterr(over='raise')

    dt = tau / dt_factor

    # Remove spikes that are not of interest (whole_train = False)
    if not whole_train:
        spike_times = spike_times[spike_times <= duration]
    else:
        duration = np.max(spike_times) + 5*tau

    if not spike_times.any():
        f = np.array([0])
        return f
    '''
    dur = np.max(spike_times) + 5 * tau
    '''

    dur = duration
    t = np.arange(0, dur, dt)
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
    return f


def new_vr(path_names, tau, dt_factor, stim_type):
    pathname = path_names[1]
    spikes = np.load(pathname + 'Calls_spikes.npy').item()

    if stim_type == 'selected':
        stims = ['naturalmothcalls/BCI1062_07x07.wav', 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav', 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav', 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav', 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav', 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav', 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl116_05x05.wav', 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav', 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav', 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav', 'naturalmothcalls/syntrichura_09x09.wav']

    if stim_type == 'series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'all_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav',
                 'callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    if stim_type == 'moth_series':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Chrostosoma_thoracicum_02.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1297_01.wav',
                 'callseries/moths/melese_PK1298_01.wav',
                 'callseries/moths/melese_PK1298_02.wav',
                 'callseries/moths/melese_PK1299_01.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_series_selected':
        stims = ['callseries/moths/A7838.wav',
                 'callseries/moths/BCI1348.wav',
                 'callseries/moths/Chrostosoma_thoracicum.wav',
                 'callseries/moths/Creatonotos.wav',
                 'callseries/moths/Eucereon_appunctata.wav',
                 'callseries/moths/Eucereon_hampsoni.wav',
                 'callseries/moths/Eucereon_maia.wav',
                 'callseries/moths/GL005.wav',
                 'callseries/moths/Hyaleucera_erythrotelus.wav',
                 'callseries/moths/Hypocladia_militaris.wav',
                 'callseries/moths/PP241.wav',
                 'callseries/moths/PP612.wav',
                 'callseries/moths/PP643.wav',
                 'callseries/moths/Saurita.wav',
                 'callseries/moths/Uranophora_leucotelus.wav',
                 'callseries/moths/carales_PK1275.wav',
                 'callseries/moths/melese_PK1300_01.wav']

    if stim_type == 'moth_single':
        stims = ['naturalmothcalls/BCI1062_07x07.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24.wav',
                 'naturalmothcalls/aclytia_gynamorpha_24x24_02.wav',
                 'naturalmothcalls/agaraea_semivitrea_06x06.wav',
                 'naturalmothcalls/agaraea_semivitrea_07x07.wav',
                 'naturalmothcalls/carales_11x11_01.wav',
                 'naturalmothcalls/carales_11x11_02.wav',
                 'naturalmothcalls/carales_12x12_01.wav',
                 'naturalmothcalls/carales_12x12_02.wav',
                 'naturalmothcalls/carales_13x13_01.wav',
                 'naturalmothcalls/carales_13x13_02.wav',
                 'naturalmothcalls/carales_19x19.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_04x04_02.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05.wav',
                 'naturalmothcalls/chrostosoma_thoracicum_05x05_02.wav',
                 'naturalmothcalls/creatonotos_01x01.wav',
                 'naturalmothcalls/creatonotos_01x01_02.wav',
                 'naturalmothcalls/elysius_conspersus_05x05.wav',
                 'naturalmothcalls/elysius_conspersus_08x08.wav',
                 'naturalmothcalls/elysius_conspersus_11x11.wav',
                 'naturalmothcalls/epidesma_oceola_05x05.wav',
                 'naturalmothcalls/epidesma_oceola_05x05_02.wav',
                 'naturalmothcalls/epidesma_oceola_06x06.wav',
                 'naturalmothcalls/eucereon_appunctata_11x11.wav',
                 'naturalmothcalls/eucereon_appunctata_12x12.wav',
                 'naturalmothcalls/eucereon_appunctata_13x13.wav',
                 'naturalmothcalls/eucereon_hampsoni_07x07.wav',
                 'naturalmothcalls/eucereon_hampsoni_08x08.wav',
                 'naturalmothcalls/eucereon_hampsoni_11x11.wav',
                 'naturalmothcalls/eucereon_obscurum_10x10.wav',
                 'naturalmothcalls/eucereon_obscurum_14x14.wav',
                 'naturalmothcalls/gl005_04x04.wav',
                 'naturalmothcalls/gl005_05x05.wav',
                 'naturalmothcalls/gl005_11x11.wav',
                 'naturalmothcalls/gl116_04x04.wav',
                 'naturalmothcalls/gl116_04x04_02.wav',
                 'naturalmothcalls/gl116_05x05.wav',
                 'naturalmothcalls/hypocladia_militaris_03x03.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09.wav',
                 'naturalmothcalls/hypocladia_militaris_09x09_02.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05.wav',
                 'naturalmothcalls/idalu_fasciipuncta_05x05_02.wav',
                 'naturalmothcalls/idalus_daga_18x18.wav',
                 'naturalmothcalls/melese_11x11_PK1299.wav',
                 'naturalmothcalls/melese_12x12_01_PK1297.wav',
                 'naturalmothcalls/melese_12x12_PK1299.wav',
                 'naturalmothcalls/melese_13x13_PK1299.wav',
                 'naturalmothcalls/melese_14x14_PK1297.wav',
                 'naturalmothcalls/neritos_cotes_07x07.wav',
                 'naturalmothcalls/neritos_cotes_10x10.wav',
                 'naturalmothcalls/neritos_cotes_10x10_02.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_06x06.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_08x08.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_09x09.wav',
                 'naturalmothcalls/ormetica_contraria_peruviana_30++.wav',
                 'naturalmothcalls/syntrichura_07x07.wav',
                 'naturalmothcalls/syntrichura_09x09.wav',
                 'naturalmothcalls/syntrichura_12x12.wav']

    if stim_type == 'moth_single_selected':
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

    if stim_type == 'all_single':
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
                 'naturalmothcalls/syntrichura_12x12.wav',
                 'batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_single':
        stims = ['batcalls/Barbastella_barbastellus_1_n.wav',
                 'batcalls/Eptesicus_nilssonii_1_s.wav',
                 'batcalls/Myotis_bechsteinii_1_n.wav',
                 'batcalls/Myotis_brandtii_1_n.wav',
                 'batcalls/Myotis_nattereri_1_n.wav',
                 'batcalls/Nyctalus_leisleri_1_n.wav',
                 'batcalls/Nyctalus_noctula_2_s.wav',
                 'batcalls/Pipistrellus_pipistrellus_1_n.wav',
                 'batcalls/Pipistrellus_pygmaeus_2_n.wav',
                 'batcalls/Rhinolophus_ferrumequinum_1_n.wav',
                 'batcalls/Vespertilio_murinus_1_s.wav']

    if stim_type == 'bats_series':
        stims = ['callseries/bats/Barbastella_barbastellus_1_n.wav',
                 'callseries/bats/Myotis_bechsteinii_1_n.wav',
                 'callseries/bats/Myotis_brandtii_1_n.wav',
                 'callseries/bats/Myotis_nattereri_1_n.wav',
                 'callseries/bats/Nyctalus_leisleri_1_n.wav',
                 'callseries/bats/Nyctalus_noctula_2_s.wav',
                 'callseries/bats/Pipistrellus_pipistrellus_1_n.wav',
                 'callseries/bats/Pipistrellus_pygmaeus_2_n.wav',
                 'callseries/bats/Rhinolophus_ferrumequinum_1_n.wav',
                 'callseries/bats/Vespertilio_murinus_1_s.wav']

    # Tags and Stimulus names
    connection, _ = tagtostimulus(path_names)
    stimulus_tags = [''] * len(stims)
    for p in range(len(stims)):
        stimulus_tags[p] = connection[stims[p]]

    cos = 1
    tau = 0.01
    probes = [[]] * len(stimulus_tags)
    for i in range(len(stimulus_tags)):
        for k in range(len(spikes[stimulus_tags[i]])):
            probes[i].append(list(spikes[stimulus_tags[i]][k]))
    # d = pymuvr.square_dissimilarity_matrix(spike_times, cos, tau, 'distance')
    d = pymuvr.distance_matrix(probes, probes, cos, tau)
    embed()
    exit()


def vanrossum_matrix2(dataset, trains, stimulus_tags, duration, dt_factor, tau, boot_sample, save_fig):

    # Randomly choose one template spike train for each call
    temp_idx = np.zeros(shape=(len(stimulus_tags), 1))
    templates = [[]] * len(stimulus_tags)
    own_probes = [[]] * len(stimulus_tags)
    probes = [[]] * len(stimulus_tags)
    for i in range(len(stimulus_tags)):
        temp_idx[i] = np.random.randint(0, len(trains[stimulus_tags[i]]), 1)
        rest_idx = [True] * len(trains[stimulus_tags[i]])
        rest_idx[np.int(temp_idx[i][0])] = False
        templates[i] = trains[stimulus_tags[i]][np.int(temp_idx[i][0])]
        own_probes[i] = np.array(trains[stimulus_tags[i]])[rest_idx]
        probes[i] = trains[stimulus_tags[i]]
    # Now compare each template with all other trains via Van Rossum Distance
    d = vanrossum_distance2(probes, temp_idx, dt_factor, tau, duration)

    embed()
    exit()

    return 0


def spike_train_distance(spike_times1, spike_times2, dt_factor, tau, plot):
    # This function computes the distance between two trains. An exponential tail is added to every event in time
    # (spike times) and then the difference between both trains is computed.

    # tau in seconds
    # dt_factor: dt = tau/dt_factor
    # spike times in seconds
    # duration in seconds
    dt = tau/dt_factor
    duration = np.max([np.max(spike_times1), np.max(spike_times2)]) + 5 * tau
    t = np.arange(0, duration, dt)
    f = np.zeros(len(t))
    g = np.zeros(len(t))

    for ti in spike_times1:
        dummy = np.heaviside(t - ti, 0) * np.exp(-(t - ti) / tau)
        f = f + dummy

    for ti in spike_times2:
        dummy = np.heaviside(t - ti, 0) * np.exp(-(t - ti) / tau)
        g = g + dummy

    # Compute Difference
    d = np.sum((f - g) ** 2) * (dt / tau)
    # f_count = np.sum(f)* (dt / tau)
    # g_count = np.sum(g) * (dt / tau)

    # print('Spike Train Difference: %s' % d)
    # print('Tau = %s' % tau)

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(t, f)
        plt.ylabel('f(t)')

        plt.subplot(2, 1, 2)
        plt.plot(t, g)
        plt.xlabel('Time [s]')
        plt.ylabel('g(t)')

        plt.show()

    return d
