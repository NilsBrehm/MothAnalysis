import matplotlib.pyplot as plt
import nixio as nix
import myfunctions as mf
import numpy as np
from IPython import embed
from scipy import signal as sg

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
datasets = ['2017-11-17-aa']

data_name = datasets[0]
pathname = "figs/" + data_name + "/"
nix_file = '/media/brehm/Data/mothdata/' + data_name + '/' + data_name + '.nix'

# Open the nix file
f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
b = f.blocks[0]

# Set Data Parameters
skips = 0
qq = 0
data_set_numbers = np.sort([117, 176, 21, 22, 24, 26, 27, 30, 32, 351, 36, 39, 44, 51, 59, 71, 88, 16, 18])
vector_strength = {}

for kk in range(1, 3):  # loop through different taus

    for sets in data_set_numbers:
        # Get tags
        # tag = sets + 1
        # tag = 10  # For TESTING only
        # tag_name = 'MothASongs-moth_song-damped_oscillation*10-' + str(tag)
        tag_name = 'MothASongs-moth_song-damped_oscillation*' + str(sets) + '-' + str(kk)

        try:
            mtag = b.multi_tags[tag_name]
        except KeyError:
            # setsmax = sets
            # print('%s data sets found' % (sets-1))
            continue

        trial_number = len(mtag.positions[:])
        if trial_number <= 1:  # if only one trial exists, skip this data
            skips += 1
            continue

        # Get metadata
        meta = mtag.metadata.sections[0]

        # Reconstruct stimulus from meta data
        stimulus_time, stimulus, gap, MetaData = mf.reconstruct_moth_song(meta)
        tau = MetaData[1, 0]
        frequency = MetaData[2, 0]
        freq = int(frequency/1000)

        try:
            x = vector_strength[freq]
        except KeyError:
            vector_strength.update({freq: {}})

        try:
            x = vector_strength[freq][tau]
        except KeyError:
            vector_strength[freq].update({tau: {}})

        # Get voltage trace
        sampling_rate = 100 * 1000  # in Hz
        volt = {}
        spt = []
        for trial in range(trial_number):
            v = mtag.retrieve_data(trial, 0)[:]
            spike_times = mf.detect_peaks(v, mph=30, mpd=50, threshold=0, edge='rising', kpsh=False,
                                          valley=True, show=False, ax=None)
            spike_times = spike_times / sampling_rate  # Now spike times are in real time (sec)
            spt = np.append(spt, spike_times)  # add all spike times in one array
            spike_count = len(spike_times)
            dummy = {0: v, 1: spike_times, 2: spike_count}
            volt.update({trial: dummy})

        # Compute vector strength
        spt_sorted = np.sort(spt)  # Sort spike times in time
        vs, phase = sg.vectorstrength(spt_sorted, gap)

        # half = stimulus_time[-1] / 2
        # half_id = np.max(np.where(spt_sorted < half))
        # spt_A = spt_sorted[0:half_id]   # From start to half time = Active Pulses
        # spt_P = spt_sorted[half_id:-1]  # From half time to end = Passive Pulses
        # vs_A, phase_A = sg.vectorstrength(spt_A, gap)
        # vs_P, phase_P = sg.vectorstrength(spt_P, gap)
        # vs, phase = sg.vectorstrength(spt_sorted, gap)
        # print('freq = %s kHz, tau = %s ms, pause = %s ms, n = %s and vsA = %s' % ((frequency/1000), (tau*1000), (gap*1000), trial_number, np.round(vs_A, 4)))

        # Store interlude data
        vector_strength[freq][(tau)].update({int(gap*1000): vs})

        # Compute PSTH (binned)
        bin_width = 2  # in ms
        bins = int(np.round(stimulus_time[-1] / (bin_width/1000)))  # in secs
        frate, bin_edges, bin_width2 = mf.psth(spt, trial_number, bins, plot=False, return_values=True)

        # --------------------------------------------------------------------------------------------------------------
        # Plotting -----------------------------------------------------------------------------------------------------
        stimulus_time = stimulus_time*1000  # now in ms

        # Raster Plot
        plt.subplot(3, 1, 1)
        for i in range(len(volt)):  # loop through trials
            sp_times = volt[i][1]
            plt.plot(sp_times*1000, np.ones(len(sp_times))+i, 'k|')
        plt.xlim(0, stimulus_time[-1])
        plt.ylabel('Trial Number')
        plt.title('gap = ' + str(gap * 1000) + ' ms, tau = ' + str(tau * 1000) + ' ms, bin = ' + str(bin_width) +
                  ' ms, freq = ' + str(frequency/1000) + ' kHz')

        # PSTH
        plt.subplot(3, 1, 2)
        plt.plot(bin_edges[:-1]*1000, frate, 'k')
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
        # plt.show()

        # Save Plot to HDD
        figname = pathname + "MothASongs_" + str(tau*1000) + '_' + str(gap*1000) + ".png"
        fig = plt.gcf()
        fig.set_size_inches(16, 12)
        fig.savefig(figname, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Progress Bar
        qq += 1
        percent = np.round(qq / (len(data_set_numbers)*2), 2)
        print('-- Plotting MothASongs: %s %%  --' % (percent * 100))

print('%s recordings were skipped (only one trial)' % skips)
# Save Data

dname = pathname + 'vs.npy'
np.save(dname, vector_strength)
print('vector strength data saved')
f.close()

# print('No vector strength data could be saved (Maybe there is none?)')
