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
# datasets = ['2018-02-09-aa']  # Calls
# datasets =['2018-02-09-aa']  # Calls
# datasets = ['2017-12-05-aa']  # FI
datasets = ['2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa', '2017-11-01-aa', '2017-11-16-aa']  # Carales FIs

FIFIELD = False
INTERVAL_MAS = False
Bootstrapping = False
INTERVAL_REC = False
GAP = False
SOUND = False

EPULSES = False
ISI = False
VANROSSUM = False
PULSE_TRAIN_ISI = False
PULSE_TRAIN_VANROSSUM = False

FI_OVERANIMALS = True


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
    data = datasets[4]
    save_plot = True
    p = "/media/brehm/Data/MasterMoth/figs/" + data + "/DataFiles/"
    th = 4
    spike_count, fi_field, fsl = mf.fifield_analysis2(data, th, plot_fi=False)
    # freqs = np.zeros(len(spike_count))
    freqs = [[]] * len(spike_count)
    i = 0
    for key in spike_count:
        freqs[i] = int(key)
        i += 1
    freqs = sorted(freqs)

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
    mf.bootstrapping_vs(datasets, nresamples, plot_histogram=True)

if SOUND:
    spikes = mf.spike_times_indexes(datasets[0], 'Calls', th_factor=4, min_dist=50, maxph=0.8, show=False, save_data=True)

if EPULSES:
    dt_factor = 100
    taus = [5, 10, 20, 30, 50]
    stim_type = 'moth_series'
    method = 'exp'
    for k in range(len(taus)):
        trains, stimulus_tags = mf.trains_to_e_pulses(datasets[0], taus[k] / 1000, 0,
                                                      dt_factor, stim_type=stim_type, whole_train=True, method=method)
    print('Finished Computing E-Pulses for taus:')
    print(taus)

if VANROSSUM:
    # Try to load e pulses from HDD
    p = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/DataFiles/"

    dt_factor = 100
    # taus = [5, 10, 20, 30, 50]
    taus = [20]
    # duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    duration = [1000]
    whole_train = True
    nsamples = 20
    stim_type = 'moth_series'
    method = 'exp'
    plot_correct = False

    # Compute VanRossum Distances
    correct = np.zeros((len(duration), len(taus)))
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
        for dur in tqdm(range(len(duration)), desc='durations'):
            mm, correct[dur, tt], distances = mf.vanrossum_matrix(datasets[0], trains, stimulus_tags, duration[dur]/1000,
                                                       dt_factor, taus[tt]/1000, boot_sample=nsamples, save_fig=False)
            # print(str((dur + 1) / len(duration) * 100) + ' % done')
    # print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))

    # Plot Percent Correct
    '''
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
    '''

if ISI:
    duration = [2000]
    #duration = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500]
    # duration = [10, 20, 40, 50, 100, 200, 400, 500, 750, 1000, 1500, 2000]
    # duration = np.arange(50, 3001, 50)
    nsamples = 20
    # profs = ['COUNT', 'ISI', 'SPIKE', 'SYNC']
    profs = ['ISI']
    plot_correct = False
    stim_type = 'moth_series'

    correct = np.zeros((len(duration), len(profs)))
    for p in range(len(profs)):
        for i in range(len(duration)):
            mm, correct[i, p], distances = mf.isi_matrix(datasets[0], duration[i]/1000, boot_sample=nsamples,
                               stim_type=stim_type, profile=profs[p], save_fig=False)
            print(str((i+1)/len(duration)*100) + ' % done')
        # print('\n' * 100)
        print('Total: ' + str((p + 1) / len(profs) * 100) + ' % done')
    if plot_correct:
        for k in range(len(profs)):
            plt.subplot(2, 2, k+1)
            plt.plot(duration, correct[:, k], 'ko-')
            plt.xlabel('Spike Train Length [ms]')
            plt.ylabel('Correct [' + profs[k] + ']')
        plt.show()

if GAP:
    tag_list = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Gap_tag_list.npy')
    sp = mf.spike_times_indexes(datasets[0], 'Gap', th_factor=2, min_dist=50, maxph=0.75, show=False)
    embed()

if PULSE_TRAIN_ISI:
    save_plot = True
    stim_type = 'callseries/moths'
    fs = 480*1000
    duration = 2000
    duration = duration / 1000
    calls, calls_names = mf.mattopy(stim_type, fs)
    d_pulses_isi, d_pulses_spikes = mf.pulse_train_matrix(calls, duration)

    d_st_isi = distances[len(distances)-1][0]
    for k in range(len(distances)-1):
        d_st_isi = d_st_isi + distances[k][0]
    d_st_isi = d_st_isi / len(distances)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(d_pulses_isi, vmin=0, vmax=1)
    plt.clim(0, 1)
    plt.xticks(np.arange(0, len(d_pulses_isi), 1))
    plt.yticks(np.arange(0, len(d_pulses_isi), 1))
    plt.xlabel('Original Call')
    plt.ylabel('Matched Call')
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.title('ISId Pulse Trains [' + str(duration*1000) + ' ms]')

    plt.subplot(1, 2, 2)
    plt.imshow(d_st_isi, vmin=0, vmax=1)
    plt.clim(0, 1)
    plt.xticks(np.arange(0, len(d_st_isi), 1))
    plt.yticks(np.arange(0, len(d_st_isi), 1))
    plt.xlabel('Original Call')
    plt.ylabel('Matched Call')
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.title('ISId Spike Trains [' + str(duration*1000) + ' ms] (boot = ' + str(nsamples) + ')')
    # plt.tight_layout()

    if save_plot:
        # Save Plot to HDD
        p = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/DataFiles/"
        figname = p + 'pulseVSspike_train_ISI_' + str(duration*1000) + 'ms_.png'
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Plot saved')
    else:
        plt.show()
    embed()

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

if PULSE_TRAIN_VANROSSUM:
    save_plot = True
    stim_type = 'callseries/moths'
    fs = 480*1000
    duration = 2000
    duration = duration / 1000
    tau = 20
    dt_factor = 100
    method = 'exp'
    e_pulses_dur = int(duration / ((tau/1000)/dt_factor))
    calls, calls_names = mf.mattopy(stim_type, fs)
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
    plt.title('VanRossum Pulse Trains [' + str(duration * 1000) + ' ms, tau=' + str(tau) + ' ms]')

    plt.subplot(1, 2, 2)
    plt.imshow(d_st_vr)
    plt.clim(0, 1)
    plt.xticks(np.arange(0, len(d_st_vr), 1))
    plt.yticks(np.arange(0, len(d_st_vr), 1))
    plt.xlabel('Original Call')
    plt.ylabel('Matched Call')
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.title('VanRossum Spike Trains [' + str(duration * 1000) + ' ms, tau=' + str(tau) + ' ms ] (boot = ' + str(nsamples) + ')')
    # plt.tight_layout()
    if save_plot:
        # Save Plot to HDD
        p = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/DataFiles/"
        figname = p + 'pulseVSspike_train_VanRossum_' + str(duration * 1000) + 'ms_' + str(tau) + 'ms.png'
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Plot saved')
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
    fields = {}
    for i in range(len(datasets)-1):
        p = "/media/brehm/Data/MasterMoth/figs/" + datasets[i] + "/DataFiles/"
        fields.update({i: np.load(p + 'fi_field.npy')})
        plt.plot(fields[i][:, 0], fields[i][:, 1], 'o-')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('dB SPL at threshold')
    plt.ylim(0, 90)
    mf.adjustSpines(plt.gca())
    plt.show()


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
