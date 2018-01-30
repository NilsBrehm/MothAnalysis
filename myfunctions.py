import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
import os
import time
import scipy.io.wavfile as wav
from scipy import signal as sg
from IPython import embed
from shutil import copyfile
import quickspikes as qs


# def read_tagged_data(m):
#     """ Reads tagged data from nix file
#
#     :param m: tag
#
#     """
#     samplingrate = 20000  # in Hz
#     meta = m.metadata.sections[0]
#     duration = meta["Duration"]
#     frequency = meta["Frequency"]
#     amplitude = m.features[4].data[1]
#     print("\n", "Stimulus: ", m.name, "\n", "Duration: ", duration, "s", "\n", "Frequency: ", frequency, "Hz", "\n",
#           "Amplitude: ", amplitude, "dB SPL")
#     m_N = len(m.positions[:])  # how many stimulus trials
#     m_extent = m.extents[1]  # how long is one stimulus
#     m_datasize = np.round(m_extent * samplingrate)  # how many datapoints will be in the dataset
#     # data = np.zeros((m_N, m_datasize))
#     # for i in range(m_N):
#     # data[i,:] = m.retrieve_data(i, recording)[:]
#     # data = np.mean(data,axis=0)
#     pos = np.delete(m.positions[:], 0)
#     index = m.features[4].data[:] == 67.0  # Look for trials with amplitude 67 db SPL
#     pos = pos[index]
#     count_all = 0
#     for k in range(len(pos)):
#         count = np.sum(m.positions[:] == pos[k])
#         count_all = count + count_all
#     dummy = m.retrieve_data(1, 0)[:].shape[0]
#     data = np.zeros((count_all, dummy))
#     spike_count = np.zeros((count_all, 1))
#     p = 0
#     for i in pos:
#         index = int(np.where(m.positions[:] == i)[0])
#         data[p, :] = m.retrieve_data(index, 0)[:]
#         spikes = m.retrieve_data(index, 1)[:] - m.positions[index]
#         spike_count[p, 0] = spikes.shape[0]
#         p = p + 1
#
#     volt = np.mean(data, axis=0)  # Mean Voltage of all trials with same amplitude
#     spike_count_mean = np.mean(spike_count)  # Mean Spike Count of all trials with same amplitude
#     # data = m.retrieve_data(i,0)[:]
#     # spikes = m.retrieve_data(i,1)[:]-m.positions[i]
#     data_size = data.shape[0]
#     t_max = data_size * (1 / samplingrate)
#     # t_step = t_max/data_size
#     time = np.linspace(0, t_max, data_size)  # (start, end, #indices)
#     y_spikes = np.ones((1, len(spikes))) * (np.max(data) + 10)
#
#     return time, data, spikes, y_spikes


def same_amp_trials(m):
    """ Returns mean spike count and mean voltage trace for trials with specific amplitude

      :param m: nix tag
      """

    # Get the size of the voltage trace
    dummy = m.retrieve_data(1, 0)[:].shape[0]
    # Get rid of the buggy first 0 in m.positions
    pos = np.delete(m.positions[:], 0)

    # Loop through all possible dB SPL values
    spike_count_mean = np.zeros((len(pos), 2))
    volt = np.zeros((len(pos), dummy))
    j = -1
    for amp in m.features[4].data[:]:
        # Get rid of the buggy first 0 in m.positions
        pos = np.delete(m.positions[:], 0)
        j += 1
        # Look for trials with desired amplitude in dB SPL
        index = m.features[4].data[:] == amp
        pos = pos[index]

        # Count trials with desired amplitude
        count_all = 0
        for k in range(len(pos)):
            count = np.sum(m.positions[:] == pos[k])
            count_all = count + count_all

        # Get the data from the desired trials
        data = np.zeros((count_all, dummy))
        spike_count = np.zeros((count_all, 1))
        p = 0
        for i in pos:
            index = int(np.where(m.positions[:] == i)[0])
            data[p, :] = m.retrieve_data(index, 0)[:]
            spikes = m.retrieve_data(index, 1)[:] - m.positions[index]
            spike_count[p, 0] = spikes.shape[0]
            p += 1
        volt[j, :] = np.mean(data, axis=0)  # Mean Voltage of all trials with same amplitude
        spike_count_mean[j, 0] = np.mean(spike_count)  # Mean Spike Count of all trials with same amplitude
        spike_count_mean[j, 1] = amp
    return volt, spike_count_mean


def read_data(data_file, nomen, wannaplot, duration=10):
    """ Reads data from nix file

    :param data_file: The path of the nix file.
    :param nomen: Name of dataset (Must be a string)
    :param wannaplot: plot = 1; no plot = 0
    :param duration: the duration of the segment that should be read. Default is 10 s.
    """
    if not os.path.exists(data_file):
        return None
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    data_array = b.data_arrays[nomen]
    print(data_array.name)
    sample_rate = 1. / data_array.dimensions[0].sampling_interval
    max_index = duration * sample_rate
    max_index = int(data_array.shape[0] if max_index > data_array.shape[0] else max_index)
    data = data_array[:max_index]
    time = np.asarray(data_array.dimensions[0].axis(max_index))

    if wannaplot == 1:
        x_axis = data_array.dimensions[0]
        x = x_axis.axis(max_index)
        # x = x_axis.axis(data_array.data.shape[0])
        # y = data_array.data
        y = data
        plt.plot(x, y)
        plt.xlabel(x_axis.label + " [" + x_axis.unit + "]")
        plt.ylabel(data_array.label + " [" + data_array.unit + "]")
        plt.title(data_array.name)
        plt.xlim(0, np.max(x))
        plt.ylim((1.1 * np.min(y), 1.1 * np.max(y)))
        plt.show()

    f.close()
    return time, data, data_array


def read_voltage(data_file, duration=10):
    """ Reads the recorded EOD from the given nix file.

    :param data_file: The path of the nix file.
    :param duration: the duration of the segment that should be read. Default is 10 s.
    :return time: numpy vector containing the time
    :return eod: numpy vector containing the eod signal.
    """
    if not os.path.exists(data_file):
        return None
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    voltage_array = b.data_arrays["V-1"]
    sample_rate = 1. / voltage_array.dimensions[0].sampling_interval
    max_index = duration * sample_rate
    max_index = int(voltage_array.shape[0] if max_index > voltage_array.shape[0] else max_index)
    voltage = voltage_array[:max_index]
    time = np.asarray(voltage_array.dimensions[0].axis(max_index))
    f.close()
    return time, voltage, voltage_array


def read_spike_events(data_file):
    """ Reads the recorded Spike events from the nix file.

    :param data_file: the path of the nix file.
    :return the event times.
    """
    if not os.path.exists(data_file):
        return None
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    events_array = b.data_arrays['Spikes-1']
    spikes = events_array[:]
    sptime = events_array.dimensions
    f.close()
    return spikes, sptime


def read_multitags(data_file):
    if not os.path.exists(data_file):
        return None
    f = nix.File.open(data_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    mtags = b.multi_tags["FICurve-sine_wave-1"]
    specdata = mtags.retrieve_data(1, 0)[:]
    f.close()
    return specdata


def get_spike_count(tag):
    """
    j = -1
    spike_count = np.zeros((tag.features[4].data[:].size,2))
    for k in tag.features[4].data[:]:
        j += 1
        spike_count[j,0] = tag.retrieve_data(j,1)[:].size
        spike_count[j,1] = k

    """

    spike_count = np.zeros((tag.features[4].data[:].size, 2))
    # spike_count = {}
    for j in range(tag.features[4].data[:].size):
        spike_count[j, 0] = tag.retrieve_data(j, 1)[:].size  # Is it possible to read all at once like (:,1) ?
        spike_count[j, 1] = tag.features[4].data[j]  # Gets Amp and stores it as well
        # spike_count.update({int(tag.features[4].data[j]) : tag.retrieve_data(j, 1)[:].size})

    return spike_count


def get_metadata(tag):
    meta = tag.metadata.sections[0]
    duration = meta["Duration"]
    frequency = meta["Frequency"]
    amplitude = tag.features[4].data[1]
    return duration, frequency, amplitude


def get_session_metadata(datasets):
    # Copies info.dat to the analysis folder
    for dat in range(len(datasets)):
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        info_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + 'info.dat'

        copyfile(info_file, pathname+'info.dat')
        print('Copied info.dat of %s' % data_name)

    return 0


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


def rect_stimulus(period, pulse_duration, stimulus_duration, total_amplitude,  plotting):
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
    sampling_rate = 800*1000
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


def get_spiketimes(tag,mtag,dataset):
    if mtag == 1:
        spike_times = tag.retrieve_data(dataset, 1)  # from multi tag get spiketimes (dataset,datatype)
    else:
        spike_times = tag.retrieve_data(1)  # from single tag get spiketimes
    return spike_times


def inter_spike_interval(spikes):
    isi = np.diff(spikes)
    return isi


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


def get_voltage(tag, mtag, dataset):
    if mtag == 1:

        volt = tag.retrieve_data(dataset, 0)  # for multi tags
    else:
        volt = tag.retrieve_data(0)  # for single tags
    return volt


# FI Field functions:

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


# def plotting_fifield_data(pathname, freq, fifield, ficurves):
#
#     # Plot FIField and save to HDD
#     plot_fifield(fifield, pathname, savefig=True)
#
#     # Plot FICurves and save to HDD
#     for i in freq:
#         plot_ficurve(ficurves[i], i, pathname, freq)
#
#     return 'Plots saved'


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
                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'],
                                               show=peak_params['show'], ax=None, maxph=peak_params['maxph'],
                                               dynamic=peak_params['dynamic'], filter_on=peak_params['filter_on'])
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

    return 0


# Spike Detection functions:

def voltage_filter(data_name, cutoff, order, ftype, filter_on=True):
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    fname = pathname + 'intervals_mas_voltage.npy'
    voltage = np.load(fname).item()
    if filter_on:
        b, a = sg.butter(order, cutoff, btype=ftype, analog=False)
        y = sg.filtfilt(b, a, voltage[0][0][0])
    else:
        y = voltage[0][0][0]
    return y


def voltage_trace_filter(voltage, cutoff, order, ftype, filter_on=True):
    if filter_on:
        b, a = sg.butter(order, cutoff, btype=ftype, analog=False)
        y = sg.filtfilt(b, a, voltage)
    else:
        y = voltage
    return y


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


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None, maxph=None, dynamic=False, filter_on=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
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
    maxph: maximal peak height
    dynamic: Dynamically calculate maximal peak height (default = False)
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

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """

    # Filter Voltage Trace
    if filter_on:
        fs = 100 * 1000
        nyqst = 0.5 * fs
        lowcut = 300
        highcut = 2000
        low = lowcut / nyqst
        high = highcut / nyqst
        x = voltage_trace_filter(x, [low, high], ftype='band', order=3, filter_on=True)

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
        if dynamic:
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


# Moth and Bat songs functions:

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
    sound = wav.read('/media/brehm/Data/MasterMoth/' + sound_file_name)
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


def soundfilestimuli_spike_detection(datasets, peak_params):
    data_name = datasets[0]
    stim_sets = [['/mothsongs/'], ['/batcalls/noisereduced/']]
    for p in stim_sets:
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + p
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
                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'], show=peak_params['show'],
                                               ax=None, maxph=peak_params['maxph'], dynamic=peak_params['dynamic'],
                                               filter_on=peak_params['filter_on'])
                    spike_times = spike_times / dt  # Now spikes are in seconds
                    sp.update({i: spike_times})

                    # Save Spike Times to HDD
                    # dname = file_name[:-12] + '_spike_times.npy'
                    # np.save(dname, sp)

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

def spike_distance_matrix(datasets):
    data_name = datasets[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + '/mothsongs/'
    stim1_name = 'Barbastella_barbastellus_1_n'
    stim2_name = 'Myotis_brandtii_1_n'

    file_name1 = stim1_name + '_spike_times.npy'
    file_name2 = stim2_name + '_spike_times.npy'

    stim1_dur = np.load(pathname + stim1_name + '_time.npy').item()
    stim2_dur = np.load(pathname + stim2_name + '_time.npy').item()

    file1 = np.load(pathname + file_name1).item()
    file2 = np.load(pathname + file_name2).item()

    duration = file1[0][-1] + 0.01
    dt = 100 * 1000
    # tau = 0.005
    tau = float(input('tau in seconds: '))
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

# PSTH

def psth(spike_times, n, bins, plot, return_values):
    # spike times must be seconds!
    # Compute histogram and calculate time dependent firing rate (binned PSTH)
    # n: number of trials
    # bins: number of bins

    hist, bin_edges = np.histogram(spike_times, bins)
    bin_width = bin_edges[1] - bin_edges[0]
    frate = hist / bin_width / n

    if plot:
        # Plot PSTH
        plt.plot(bin_edges[:-1], frate, 'k')

    if return_values:
        return frate, bin_edges, bin_width
    else:
        return 0


def raster_plot(sp_times, stimulus_time):
    for k in range(len(sp_times)):
        plt.plot(sp_times[k]*1000, np.ones(len(sp_times[k])) + k, 'k|')

    plt.xlim(0, stimulus_time[-1]*1000)
    plt.ylabel('Trial Number')

    return 0

# Interval MothASongs functions:

def get_moth_intervals_data(datasets):
    # Get voltage trace from nix file for Intervals MothASongs Protocol
    # data_set_numbers has to fit the the data names in the nix file!
    # Returns voltage trace, stimulus time, stimulus amplitude, gap and meta data for every single trial
    # datasets can be a list of many recordings

    idx2 = 0

    for dat in range(len(datasets)):
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + data_name + '.nix'

        # Open the nix file
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        b = f.blocks[0]
        # Set Data Parameters
        skips = 0
        idx = 0
        # Be careful not to load data that is not from the interval protocol
        data_set_numbers = ['351-1','176-1','117-1','88-1','71-1','59-1','51-1','44-1','39-1','36-1','32-1','30-1',
                            '27-1','26-1','24-1','117-2','88-2','71-2','59-2','51-2','44-2','39-2','36-2','32-2',
                            '30-2','27-2','26-2','24-2','22-1','21-1']
        voltage = {}
        for sets in data_set_numbers:
            tag_name = 'MothASongs-moth_song-damped_oscillation*' + str(sets)
            try:
                mtag = b.multi_tags[tag_name]
            except KeyError:
                print('%s not found' % str(sets))
                continue

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
            voltage.update({idx: volt})
            idx += 1

            # Progress Bar
            percent = np.round(idx / len(data_set_numbers), 2)
            print('-- Get Intervals MothASongs Data: %s %%  --' % (percent * 100))

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

        print('Intervals MothASongs Data saved for %s!' % data_name)

        idx2 += 1
        percent2 = np.round(idx2 / len(datasets), 2)
        print('-- Get Total: %s %%  --' % (percent2 * 100))

    print('All data sets done!')
    f.close()
    return voltage


def moth_intervals_spike_detection(datasets, peak_params, show_trial_detection):
    # Load data
    for dat in range(len(datasets)):
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        fname = pathname + 'intervals_mas_voltage.npy'
        voltage = np.load(fname).item()
        sampling_rate = 100 * 1000  # in Hz
        # Now detect spikes in each trial and update input data
        for i in voltage:
            spikes = []
            # for trial in voltage[i]:
            for trial in range(voltage[i]['trials']):
                x = voltage[i][trial][0]
                if show_trial_detection and trial == 0:
                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'], show=True,
                                               ax=None, maxph=peak_params['maxph'], dynamic=peak_params['dynamic'],
                                               filter_on=peak_params['filter_on'])
                    print('set number: %s' % i)
                else:
                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'],
                                               show=peak_params['show'], ax=None, maxph=peak_params['maxph'],
                                               dynamic=peak_params['dynamic'], filter_on=peak_params['filter_on'])

                spike_times = spike_times / sampling_rate  # Now spike times are in real time (seconds)
                spikes = np.append(spikes, spike_times)  # Put spike times of each trial in one long array
                spike_count = len(spike_times)
                voltage[i][trial].update({'spike_times': spike_times, 'spike_count': spike_count})
            voltage[i].update({'all_spike_times': spikes})

        # Save detected Spikes to HDD
        if not show_trial_detection:
            dname = pathname + 'intervals_mas_spike_times.npy'
            np.save(dname, voltage)
    print('Spike Detection done')
    if show_trial_detection:
        print('Data was NOT saved since trial display was TRUE')
    return 0


def moth_intervals_analysis(datasets):
    for dat in range(len(datasets)):
        # Load data
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
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
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
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
        bins = int(np.round(stimulus_time[-1] / (bin_width / 1000)))  # in secs
        frate, bin_edges, bin_width2 = psth(spt, trial_number, bins, plot=False, return_values=True)

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


# Rect Intervals

def get_rect_intervals_data(datasets):

    for dat in range(len(datasets)):  # Loop through all recording sessions in the list
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + data_name + '/' + data_name + '.nix'

        # Open the nix file
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        b = f.blocks[0]

        # Read all tag names in nix file
        i = 0
        protocol_list = []
        try:
            while True:
                protocol_list = np.append(protocol_list, b.tags[i].name)
                i += 1
        except KeyError:
            print('--- Done looking for all tags in recording ---')
        idx = np.char.startswith(protocol_list, 'SingleStimulus_')
        all_protocols = protocol_list[idx]

        # Collect data
        voltage = {}
        stim = {}
        for protocol_number in range(len(all_protocols)):  # Loop through all tags = protocols
            tag_name = all_protocols[protocol_number]
            tag = b.tags[tag_name]

            # Meta Data
            meta = tag.metadata
            macro = meta.sections[0].sections[0].props[1].name  # Get the name of the macro

            if macro != 'PulseIntervalsRect':  # Check if it is the desired macro protocol
                continue

            # Collect meta data for stimulus
            period = meta.sections[0].sections[1].props['period'].values[0].value
            pulse_duration = meta.sections[0].sections[1].props['pulseduration'].values[0].value
            stimulus_duration = meta.sections[0].sections[1].props['duration'].values[0].value
            amplitude = meta.sections[0].sections[1].props['amplitude'].values[0].value
            intensity = meta.sections[0].sections[1].props['intensity'].values[0].value
            total_amplitude = amplitude+intensity

            # Reconstruct stimulus
            stimulus_time, stimulus = rect_stimulus(period, pulse_duration, stimulus_duration, total_amplitude, plotting=False)

            # Get voltage trace
            voltage.update({protocol_number: {'voltage': tag.retrieve_data(0)[:]}})
            stim.update({protocol_number: {'stimulus': stimulus, 'stimulus_time': stimulus_time,
                                           'pulse_duration': pulse_duration, 'gap': (period-pulse_duration)}})

        # Save Data to HDD
        dname = pathname + 'intervals_rect_voltage.npy'
        np.save(dname, voltage)
        dname2 = pathname + 'intervals_rect_stimulus.npy'
        np.save(dname2, stim)

        f.close()  # Close the nix file
        print('Data Gathering done')
    return 0


def rect_intervals_spike_detection(datasets, peak_params, show_trial_detection):
    for dat in range(len(datasets)):  # Loop through all recording sessions in the list
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        filename = pathname + 'intervals_rect_voltage.npy'

        # Load Voltage Data
        sampling_rate = 100*1000
        voltage = np.load(filename).item()
        spikes = {}
        for i in voltage:
            x = voltage[i]['voltage']
            if show_trial_detection:
                spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                           edge='rising', kpsh=False, valley=peak_params['valley'], show=True,
                                           ax=None, maxph=peak_params['maxph'], dynamic=peak_params['dynamic'],
                                           filter_on=peak_params['filter_on'])
            else:
                spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                           edge='rising', kpsh=False, valley=peak_params['valley'],
                                           show=peak_params['show'], ax=None, maxph=peak_params['maxph'],
                                           dynamic=peak_params['dynamic'], filter_on=peak_params['filter_on'])

            spike_times = spike_times/sampling_rate  # Now spike times are in seconds
            spikes.update({i: spike_times})

        # Save Data to HDD
        if not show_trial_detection:
            dname = pathname + 'intervals_rect_spike_times.npy'
            np.save(dname, spikes)
        print('Spike Detection done')
    if show_trial_detection:
        print('Data was NOT saved since trial display was TRUE')
    return spikes


def rect_intervals_cut_trials(datasets):
    # Load spike times
    for dat in range(len(datasets)):  # Loop through all recording sessions in the list
        data_name = datasets[dat]
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
        filename = pathname + 'intervals_rect_spike_times.npy'
        spike_times = np.load(filename).item()
        limits = np.arange(0.5, 0.5*20, 1)  # end of trials, after this there is always a 500 ms pause
        all_trial_spike_times ={}
        for sp in spike_times:
            trial_spike_times = {}
            for trials in range(len(limits)):
                x = spike_times[sp][np.logical_and(spike_times[sp] < limits[trials], spike_times[sp] > limits[trials]-0.5)]
                x = x - (limits[trials]-0.5)  # Shift spike times to zero start = stimulus onset!
                trial_spike_times.update({trials: x})

            all_trial_spike_times.update({sp: {'trials': trial_spike_times}})

        # Save Data to HDD
        dname = pathname + 'intervals_rect_trials.npy'
        np.save(dname, all_trial_spike_times)
        print('Cutting out single trials done')
    return 0


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


# Bootsrapping
def bootstrapping_vs(datasets, nresamples, plot_histogram):
    # Load data
    data_name = datasets[0]
    pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + "/"
    fname = pathname + 'intervals_mas_spike_times.npy'
    fname2 = pathname + 'intervals_mas_vs.npy'
    spike_times = np.load(fname).item()
    vector_strength = np.load(fname2).item()
    resamples = {}
    # nresamples = 1000  # Number of resamples

    # Create Directory
    directory = os.path.dirname(pathname + 'vs/')
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    text_file_name = pathname + 'vs/bootstrap_vs.txt'

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


# Spike Train Distance

def spike_train_distance(spike_times1, spike_times2, dt, duration, tau, plot):
    # This function computes the distance between two trains. An exponential tail is added to every event in time
    # (spike times) and then the difference between both trains is computed.

    # tau in seconds
    # sampling rate dt in Hz
    # spike times in seconds
    # duration in seconds

    t = np.arange(0, duration, 1 / dt)
    f = np.zeros(len(t))
    g = np.zeros(len(t))

    for ti in spike_times1:
        dummy = np.heaviside(t - ti, 0) * np.exp(-(t - ti) / tau)
        f = f + dummy

    for ti in spike_times2:
        dummy = np.heaviside(t - ti, 0) * np.exp(-(t - ti) / tau)
        g = g + dummy

    # Compute Difference
    d = np.sum((f - g) ** 2) / (tau * dt)  # For this formula tau must be in samples!
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

def quickspikes_detection(datasets):
    data_name = datasets[0]
    stim_set = [['/mothsongs/'], ['/batcalls/noisereduced/']]

    for p in stim_set[1]:
        pathname = "/media/brehm/Data/MasterMoth/figs/" + data_name + p
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
                peak_params = {'mph': 50, 'mpd': 100, 'valley': False, 'show': False, 'maxph': 300, 'dynamic': False,
                               'filter_on': False}

                for j in range(trials):
                    x = voltage[j]
                    plot_title = ' - peaks'
                    if abs(np.max(x)) < abs(np.min(x)):
                        x = -x
                        plot_title = ' - valleys'

                    spike_times = detect_peaks(x, mph=peak_params['mph'], mpd=peak_params['mpd'], threshold=0,
                                               edge='rising', kpsh=False, valley=peak_params['valley'],
                                               show=peak_params['show'],
                                               ax=None, maxph=peak_params['maxph'], dynamic=peak_params['dynamic'],
                                               filter_on=peak_params['filter_on'])
                    reldet = qs.detector(2.0, 50)
                    reldet.scale_thresh(x.mean(), x.std())
                    times = reldet.send(x)
                    peaks = x[times]
                    peaks2 = x[spike_times]
                    plt.plot(x)
                    plt.plot(times, peaks, 'ro')
                    plt.plot(spike_times, peaks2, 'bo')
                    pt = 'Data #' + str(k) + ' - ' + str(j) + plot_title
                    plt.title(pt)
                    plt.show()
    return 0

# for i in voltage: print(str(voltage[i][0][1]) + ' --- '  + str(voltage[i]['vs']) + ' --- ' + str(voltage[i]['vs_phase']) + ' ---  ' + str(voltage[i]['gap']))
# for i in range(34): print(b.multi_tags[i].name)
