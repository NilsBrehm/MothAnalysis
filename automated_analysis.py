from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import seaborn as sns
import pickle
import matplotlib
from matplotlib.colors import LogNorm
import scipy.io.wavfile as wav

start_time = time.time()

GET_DATA = False
SPIKE_DETECTION = False
CALLS = True

# Compute Van Rossum Distance
EPULSES = True
VANROSSUM = True

# Compute other Distances
ISI = True
DISTANCE_RATIOS = True

# datasets = ['2017-11-03-aa', '2017-11-14-aa', '2017-11-16-aa', '2017-11-17-aa', '2017-11-25-aa', '2017-11-25-ab',
#             '2017-11-27-aa', '2017-12-01-aa', '2017-12-05-ab', '2018-01-26-ab', '2018-02-16-aa', '2017-11-29-aa',
#             '2017-12-04-aa', '2018-02-15-aa']
datasets = ['2018-02-16-aa', '2018-02-15-aa']

stims = ['moth_single_selected', 'moth_series_selected', 'all_series', 'all_single']
stims_l = ['single', 'series', 'series', 'single']

for ss in tqdm(range(len(stims)), desc='Total'):
    for dd in tqdm(range(len(datasets)), desc='data sets'):
        data_name = datasets[dd]
        path_names = mf.get_directories(data_name=data_name)
        print(data_name)
        p = path_names[1]

        if GET_DATA:
            mf.get_voltage_trace(path_names, 'SingleStimulus-file-', 'Calls', multi_tag=True, search_for_tags=True)
        if SPIKE_DETECTION:
            mf.spike_times_calls(path_names, 'Calls', show=False, save_data=True, th_factor=3, filter_on=True, window=None)

        # Settings for Call Analysis ===================================================================================
        # General Settings
        stim_type = stims[ss]
        stim_length = stims_l[ss]
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
        taus = list(np.concatenate([np.arange(1, 21, 1), np.arange(30, 105, 5), np.arange(200, 1000, 55)]))
        taus.append(1000)

        # ==============================================================================================================

        if EPULSES:
            datasets = [datasets[dd]]
            for i in range(len(datasets)):
                data_name = datasets[i]
                path_names = mf.get_directories(data_name=data_name)
                print(data_name)
                method = 'exp'
                r = Parallel(n_jobs=-2)(delayed(mf.trains_to_e_pulses)(path_names, taus[k] / 1000, dt, stim_type=stim_type
                                                                       , method=method) for k in range(len(taus)))
                # print('Converting done')

        if VANROSSUM:
            # Try to load e pulses from HDD
            method = 'exp'

            # Compute VanRossum Distances
            correct = np.zeros((len(duration), len(taus)))
            dist_profs = {}
            matches = {}
            groups = {}
            for tt in tqdm(range(len(taus)), desc='taus', leave=False):
                try:
                    # Load e-pulses if available:
                    trains = np.load(p + 'e_trains_' + str(taus[tt]) + '_' + stim_type + '.npy').item()
                    stimulus_tags = np.load(p + 'stimulus_tags_' + str(taus[tt]) + '_' + stim_type + '.npy')
                    # print('Loading e-pulses from HDD done')
                except FileNotFoundError:
                    # Compute e pulses if not available
                    print('Could not find e-pulses, will try to compute it on the fly')

                distances = [[]] * len(duration)
                mm = [[]] * len(duration)
                gg = [[]] * len(duration)
                # Parallel loop through all durations for a given tau
                r = Parallel(n_jobs=-2)(
                    delayed(mf.vanrossum_matrix)(data_name, trains, stimulus_tags, duration[dur] / 1000, dt, taus[tt] / 1000,
                                                 boot_sample=nsamples, save_fig=False) for dur in range(len(duration)))
                # mf.vanrossum_matrix2(data_name, trains, stimulus_tags, duration/1000, dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=True)
                # mm_mean, correct_matches, distances_per_boot, gg_mean = mf.vanrossum_matrix(data_name, trains, stimulus_tags, duration[-1]/1000, dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=True)

                # Put values from parallel loop into correct variables
                for q in range(len(duration)):
                    mm[q] = r[q][0]
                    gg[q] = r[q][3]
                    correct[q, tt] = r[q][1]
                    distances[q] = r[q][2]
                dist_profs.update({taus[tt]: distances})
                matches.update({taus[tt]: mm})
                groups.update({taus[tt]: gg})
            # Save to HDD
            np.save(p + 'VanRossum_' + stim_type + '.npy', dist_profs)
            np.save(p + 'VanRossum_correct_' + stim_type + '.npy', correct)
            np.save(p + 'VanRossum_matches_' + stim_type + '.npy', matches)
            np.save(p + 'VanRossum_groups_' + stim_type + '.npy', groups)
            # print('VanRossum Distances done')

        if ISI:
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
                r = Parallel(n_jobs=-2)(delayed(mf.isi_matrix)(path_names, duration[i] / 1000, boot_sample=nsamples,
                                                               stim_type=stim_type, profile=profs[p], save_fig=save_fig) for i
                                        in range(len(duration)))

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
            np.save(path_save + 'distances_' + stim_type + '.npy', dist_profs)
            np.save(path_save + 'distances_correct_' + stim_type + '.npy', correct)
            np.save(path_save + 'distances_rand_correct_' + stim_type + '.npy', rand_correct)
            np.save(path_save + 'distances_matches_' + stim_type + '.npy', matches)

        if DISTANCE_RATIOS:
            # Try to load e pulses from HDD
            method = 'exp'
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

            import pyspike as spk

            dur = np.array(duration) / 1000

            results = np.zeros(shape=(len(dur), 5))
            results_sync = np.zeros(shape=(len(dur), 5))
            for j in tqdm(range(len(dur)), desc='Distances'):
                edges = [0, dur[j]]
                d = np.zeros(len(calls))
                d_sync = np.zeros(len(calls))
                sp = [[]] * len(calls)
                for k in range(len(calls)):
                    spike_times = [[]] * len(spikes[stimulus_tags[k]])
                    for i in range(len(spikes[stimulus_tags[k]])):
                        spike_times[i] = spk.SpikeTrain(list(spikes[stimulus_tags[k]][i]), edges)
                    sp[k] = spike_times
                    d[k] = abs(spk.isi_distance(spike_times, interval=[0, dur[j]]))
                    d_sync[k] = spk.spike_sync(spike_times, interval=[0, dur[j]])
                    # d[k] = abs(spk.spike_distance(spike_times, interval=[0, dur[j]]))
                sp = np.concatenate(sp)
                over_all = abs(spk.isi_distance(sp, interval=[0, dur[j]]))
                over_all_sync = spk.spike_sync(sp, interval=[0, dur[j]])
                # over_all = abs(spk.spike_distance(sp, interval=[0, dur[j]]))
                ratio = over_all / np.mean(d)
                diff = over_all - np.mean(d)
                ratio_sync = over_all_sync / np.mean(d_sync)
                diff_sync = over_all_sync - np.mean(d_sync)

                results[j, :] = [np.mean(d), np.std(d), over_all, ratio, diff]
                results_sync[j, :] = [np.mean(d_sync), np.std(d_sync), over_all_sync, ratio_sync, diff_sync]

            # Save to HDD
            np.save(p + 'ISI_Ratios_' + stim_type + '.npy', results)
            np.save(p + 'SYNC_Ratios_' + stim_type + '.npy', results_sync)
            # print('Ratios saved')

print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))


