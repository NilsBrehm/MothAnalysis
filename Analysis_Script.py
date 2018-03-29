from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm

start_time = time.time()
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa']
datasets =['2018-02-09-aa']

FIFIELD = False
INTERVAL_MAS = False
Bootstrapping = False
INTERVAL_REC = False
SOUND = False
EPULSES = False
VANROSSUM = True
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

if EPULSES:
    dt_factor = 100
    taus = [5, 10, 20, 30, 50]
    method = 'exp'
    for k in range(len(taus)):
        trains, stimulus_tags = mf.trains_to_e_pulses(datasets[0], taus[k] / 1000, 0,
                                                      dt_factor, stim_type='series', whole_train=True, method=method)
    print('Finished Computing E-Pulses for taus:')
    print(taus)

if VANROSSUM:
    # Try to load e pulses from HDD
    p = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/DataFiles/"

    dt_factor = 100
    taus = [5, 10, 20, 30, 50]
    duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    # duration = [100, 500]
    whole_train = True
    nsamples = 5
    stim_type = 'series'
    plot_correct = True

    # Compute VanRossum Distances
    correct = np.zeros((len(duration), len(taus)))
    for tt in tqdm(range(len(taus)), desc='taus', leave=False):
        try:
            # Load e-pulses if available:
            trains = np.load(p + 'e_trains_' + str(taus[tt]) + '_' + stim_type + '.npy').item()
            stimulus_tags = np.load(p + 'stimulus_tags_' + str(taus[tt]) + '_' + stim_type + '.npy')
        except FileNotFoundError:
            # Compute e pulses if not available
            print('Could not find e-pulses, will try to compute it on the fly')
            trains, stimulus_tags = mf.trains_to_e_pulses(datasets[0], taus[tt]/1000, np.max(duration)/1000, dt_factor,
                                                          stim_type='series', whole_train=whole_train, method='rect')
        for dur in tqdm(range(len(duration)), desc='durations'):
            mm, correct[dur, tt] = mf.vanrossum_matrix(datasets[0], trains, stimulus_tags, duration[dur]/1000,
                                                       dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=False)
            # print(str((dur + 1) / len(duration) * 100) + ' % done')
    print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))

    # Plot Percent Correct
    print('Ready to Plot?')
    a = input('Press Enter to Continue or type something to halt')
    if not a == '':
        embed()

    for k in range(len(taus)):
        plt.subplot(3, 2, k+1)
        plt.plot(duration, correct[:, k], 'ko-')
        plt.xlabel('Spike Train Length [ms]')
        plt.ylabel('Correct [' + str(taus[k]) + ']')
    if plot_correct:
        # Save Plot to HDD
        figname = p + 'PercentCorrect_VanRossum.png'
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

if ISI:
    # duration = 1000
    duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    # duration = [10, 20, 40, 50, 100, 200, 400, 500, 750, 1000, 1500, 2000]
    # duration = np.arange(50, 3001, 50)
    nsamples = 5
    profs = ['COUNT', 'ISI', 'SPIKE', 'SYNC']
    # profs = ['COUNT']

    correct = np.zeros((len(duration), len(profs)))
    for p in range(len(profs)):
        for i in range(len(duration)):
            mm, correct[i, p] = mf.isi_matrix(datasets[0], duration[i]/1000, boot_sample=nsamples,
                               stim_type='series', profile=profs[p], save_fig=True)
            print(str((i+1)/len(duration)*100) + ' % done')
        # print('\n' * 100)
        print('Total: ' + str((p + 1) / len(profs) * 100) + ' % done')

    for k in range(len(profs)):
        plt.subplot(2, 2, k+1)
        plt.plot(duration, correct[:, k], 'ko-')
        plt.xlabel('Spike Train Length [ms]')
        plt.ylabel('Correct [' + profs[k] + ']')
    plt.show()
    embed()

if GAP:
    tag_list = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Gap_tag_list.npy')
    sp = mf.spike_times_indexes(datasets[0], 'Gap', th_factor=2, min_dist=50, maxph=0.75, show=False)
    embed()

'''
if VANROSSUM:
    dt_factor = 100
    # taus = [0.5, 1, 2, 5, 10, 20]  # in ms
    taus = [5, 10]
    duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    # duration = [50, 100, 250, 500, 750, 1000, 1500, 2000]
    nsamples = 5
    correct = np.zeros((len(duration), len(taus)))
    for tt in range(len(taus)):
        for dur in range(len(duration)):
            mm, correct[dur, tt] = mf.vanrossum_matrix(datasets[0], taus[tt]/1000, duration[dur]/1000, dt_factor,
                                                       boot_sample=nsamples, stim_type='series', save_fig=False)
            print(str((dur + 1) / len(duration) * 100) + ' % done')
    print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))

    # Plot Percent Correct
    print('Ready to Plot?')
    embed()
    for k in range(len(taus)):
        plt.subplot(2, 2, k+1)
        plt.plot(duration, correct[:, k], 'ko-')
        plt.xlabel('Spike Train Length [ms]')
        plt.ylabel('Correct [' + str(taus[k]) + ']')
    plt.show()
    embed()
    # mf.tagtostimulus(datasets[0])
'''

print('Analysis done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
