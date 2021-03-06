from IPython import embed
import myfunctions as mf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io.wavfile as wav
import os
from scipy import signal as sg

start_time = time.time()
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
datasets = ['2018-02-20-aa']

PlotRectIntervals = False
PlotMothIntervals = False
PlotVS = True
PlotVSRect = False
PlotFICurves = False
PlotFIField = False
PlotSoundRasterPlot = False
CallsRaster = False

data_name = datasets[0]

for i in range(len(datasets)):  # Loop through all recordings in the list above
    data_name = datasets[i]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/DataFiles/"
    try:
        if PlotRectIntervals:
            mf.rect_intervals_plot(data_name)

        if PlotMothIntervals:
            mf.moth_intervals_plot(data_name, 10, 50)

        if PlotVSRect:
            # # Plot Vector Strength: # #
            # Load vs
            vs = np.load(pathname + 'intervals_rect_vs.npy').item()

        if PlotVS:
            # # Plot Vector Strength: # #
            # Load vs
            vs = np.load(pathname + 'intervals_mas_vs.npy').item()
            v_strength_01, v_strength_04 = [], []
            gap_01, gap_04 = [], []

            # Plot gap vs vs
            for k in vs:
                if vs[k]['tau'] == 0.0004:  # only look for tau = 0.1 ms
                    v_strength_04 = np.append(v_strength_04, vs[k]['vs'])
                    gap_04 = np.append(gap_04, vs[k]['gap'])
                elif vs[k]['tau'] == 0.0001:
                    v_strength_01 = np.append(v_strength_01, vs[k]['vs'])
                    gap_01 = np.append(gap_01, vs[k]['gap'])

            idx01 = np.argsort(gap_01)
            idx04 = np.argsort(gap_04)

            plt.subplot(2,1,1)
            plt.plot(gap_01[idx01], v_strength_01[idx01], 'k--o')
            plt.title('tau = 0.1 ms')
            plt.ylabel('Vector Strength')

            # plt.subplot(2, 1, 2)
            # plt.plot(gap_04[idx04], v_strength_04[idx04], 'k--o')
            # plt.title('tau = 0.4 ms')
            # plt.ylabel('Vector Strength')
            # plt.xlabel('gap [s]')

            # Save Plot to HDD
            figname = pathname + "VS_50kHz.png"
            fig = plt.gcf()
            fig.set_size_inches(16, 12)
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close(fig)

        if PlotFICurves:
            # # Plot FIField and FICurves: # #
            # Load FI Curve Data for given frequency and save the plot to HDD
            # Create Directory
            pathname_fi = pathname + 'fi/'
            directory = os.path.dirname(pathname_fi)
            if not os.path.isdir(directory):
                os.mkdir(directory)  # Make Directory

            freqs_used = np.unique(np.load(pathname + 'frequencies.npy'))/1000
            for f in freqs_used:
                fi = np.load(pathname + 'FICurve_' + str(f) + '.npz')
                mf.plot_ficurve(fi['amplitude_sorted'], fi['mean_spike_count'], fi['std_spike_count'], f, fi['spike_threshold'], pathname_fi, savefig=True)

        if PlotFIField:
            pathname_fi = pathname + 'fi/'
            directory = os.path.dirname(pathname_fi)
            if not os.path.isdir(directory):
                os.mkdir(directory)  # Make Directory
            fifield = np.load(pathname + 'FIField_plotdata.npy')
            mf.plot_fifield(fifield, pathname_fi, savefig=True)

        if PlotSoundRasterPlot:
            # Load Stimulus Sound File
            # stim_name = ['Barbastella_barbastellus_1_n', 'Myotis_bechsteinii_1_n', 'Myotis_brandtii_1_n', 'Myotis_nattereri_1_n', 'Nyctalus_leisleri_1_n', 'Pipistrellus_pipistrellus_1_n']
            stim_name = ['carales_11x11_01']
            # stim_path = '/media/brehm/Data/MasterMoth/batcalls/noisereduced/'
            stim_path = '/media/brehm/Data/MasterMoth/stimuli/naturalmothcalls/'
            pathname_bats = pathname + 'naturalmothcalls/'
            directory = os.path.dirname(pathname_bats)
            if not os.path.isdir(directory):
                os.mkdir(directory)  # Make Directory

            for i in range(len(stim_name)):
                stimulus = wav.read(stim_path + stim_name[i] + '.wav')
                stimulus_time = np.linspace(0, len(stimulus[1]) / stimulus[0], len(stimulus[1]))  # in s
                fs = stimulus[0]
                # Load Spike Times
                # file_name = pathname + 'batcalls/noisereduced/' + stim_name[i] + '_spike_times.npy'
                file_name = pathname + 'naturalmothcalls/' + stim_name[i] + '_spike_times.npy'
                spike_times = np.load(file_name).item()

                # Plot
                plt.subplot(3, 1, 1)
                mf.raster_plot(spike_times, stimulus_time)
                plt.title(stim_name[i])
                plt.subplot(3, 1, 2)
                plt.plot(stimulus_time, stimulus[1], 'k')
                plt.xlim(0, np.max(stimulus_time))
                plt.subplot(3, 1, 3)
                plt.specgram(stimulus[1], NFFT=256, Fs=fs, Fc=0, detrend=mlab.detrend_none,
                        window=mlab.window_hanning, noverlap=250,
                        cmap='hot', xextent=None, pad_to=None, sides='default',
                        scale_by_freq=True, mode='default', scale='default')
                plt.ylim(0, 100000)
                plt.clim(-20, 100)
                # plt.show()

                # Save Plot to HDD
                figname = pathname_bats + stim_name[i] +".png"
                fig = plt.gcf()
                fig.set_size_inches(16, 12)
                fig.savefig(figname, bbox_inches='tight', dpi=300)
                plt.close(fig)

        if CallsRaster:
            # Load Data
            file_path = pathname + 'DataFiles/'
            stimulus_path = '/media/brehm/Data/MasterMoth/stimuli/'
            spike_times = np.load(file_path + 'Calls_spikes.npy').item()
            tag_list = np.load(file_path + 'Calls_tag_list.npy')

            # Get mtags and meta data
            f, mtags = mf.get_metadata(datasets[0], 'SingleStimulus-file-', 'Calls')

            for j in range(len(tag_list)):
                # Get MetaData
                stimulus_name = mtags[tag_list[j]].metadata.sections[0][2]

                # Get Stimulus (Sound File): stimulus[0] = fs, stimulus[1] = audio
                stimulus = wav.read(stimulus_path + stimulus_name)
                stimulus_time = np.linspace(0, len(stimulus[1]) / stimulus[0], len(stimulus[1]))  # in s

                # Put trials in one vector for PSTH
                sp = []
                trials = len(spike_times[tag_list[j]])
                for k in range(trials):
                    sp = np.append(sp, spike_times[tag_list[j]][k])

                # PLOT -------------------------------------------------------
                steps = 0.01  # x-axis step size (time steps in seconds)
                x_left_end = -0.01
                # Stimulus
                plt.subplot(3, 1, 1)
                stimulus_norm = stimulus[1] / np.max(abs(stimulus[1]))
                plt.plot(stimulus_time, stimulus_norm, 'k')
                plt.xticks(np.arange(0, stimulus_time[-1], steps))
                plt.xlim(x_left_end, stimulus_time[-1])
                plt.yticks([-1, -0.5, 0, 0.5, 1])
                plt.ylim(-1, 1)
                plt.ylabel('Amplitude')

                # Raster Plot
                plt.subplot(3, 1, 2)
                mf.raster_plot(spike_times[tag_list[j]], stimulus_time, steps)
                plt.xticks(np.arange(0, stimulus_time[-1], steps))
                plt.xlim(x_left_end, stimulus_time[-1])

                # Plot PSTH
                plt.subplot(3, 1, 3)
                bin_size = 2  # in ms
                frate, bin_edges = mf.psth(sp, trials, bin_size/1000, plot=False, return_values=True, separate_trials=False)
                plt.plot(bin_edges[:-1], frate, 'k')
                plt.xlabel('Time [s] ' + '(bin size = ' + str(bin_size) + ' ms)')
                plt.ylabel('Mean Firing Rate [spikes/s]')
                plt.ylim(0, np.max(frate)+100)
                plt.xticks(np.arange(0, stimulus_time[-1], steps))
                plt.xlim(x_left_end, stimulus_time[-1])

                # Save Plot to HDD
                # s_name = stimulus_name[stimulus_name.find('/')+1:-4]
                s_name = stimulus_name[:-4]
                figname = pathname + s_name + ".png"
                fig = plt.gcf()
                fig.set_size_inches(16, 12)
                fig.savefig(figname, bbox_inches='tight', dpi=300)
                plt.close(fig)

            f.close()

    except FileNotFoundError:
        print('File not found')
        continue

print("--- Plotting took %s minutes ---" % np.round((time.time() - start_time) / 60, 2))
