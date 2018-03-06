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
import quickspikes as qs
import itertools as itertools

# ----------------------------------------------------------------------------------------------------------------------
# PEAK DETECTION


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


def indexes(y, th_factor=2, min_dist=50, maxph=0.8):
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

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    thres = th_factor * np.median(abs(y) / 0.6745)
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

    return peaks


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


# ----------------------------------------------------------------------------------------------------------------------
# NIX Functions


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


def get_voltage_trace(dataset, tag, protocol_name, multi_tag, search_for_tags):
    """Get Voltage Trace from nix file.

    Notes
    ----------
    This function reads out the voltage traces for the given tag names stored in the nix file

    Parameters
    ----------
    dataset :       Data set name (string)
    tag:            Tag name (string)
    protocol_name:  protocol name (string)
    search_for_tags: search for all tags containing the string in tag. If set to false, the list in 'tag' is used.
    multi_tag: If true then function treats tags as multi tags and looks for all trials.

    Returns
    -------
    voltage: Saves Voltage Traces to HDD in a .npy file (dict)

    """

    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + dataset + '/' + dataset + '.nix'
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


def list_protocols(dataset, protocol_name, tag_name):
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + dataset + '/' + dataset + '.nix'

    try:
        f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
        print('".nix" extension found')
        print(dataset)
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

    tag_list = [t.name for t in b.tags if tag_name[0] in t.name]
    mtag_list = [t.name for t in b.multi_tags if tag_name[1] in t.name]
    target = []
    mtarget = []

    # Create Text File:
    text_file_name = file_pathname + 'stimulus_protocols.txt'
    text_file_name2 = file_pathname + 'stimulus_songs.txt'

    try:
        text_file = open(text_file_name, 'r+')
        text_file2 = open(text_file_name2, 'r+')
        text_file.truncate(0)
        text_file2.truncate(0)
    except FileNotFoundError:
        print('create new txt files')

    # Tags:
    p = [''] * len(tag_list)
    try:
        for i in range(len(tag_list)):
            p[i] = b.tags[tag_list[i]].metadata.sections[0].sections[0].props[1].name
            if p[i] == protocol_name:
                target.append(tag_list[i])

        # Write to txt file
        with open(text_file_name, 'a') as text_file:
            text_file.write(dataset + '\n')
            text_file.write(tag_name[0] + '\n\n')
            for k in range(len(p)):
                text_file.write(p[k] + '\n')
    except KeyError:
        print('No Protocols found')

    # Multi Tags:
    mp = [''] * len(mtag_list)
    try:
        for i in range(len(mtag_list)):
            mp[i] = b.multi_tags[mtag_list[i]].metadata.sections[0][2]  # This is the sound file name
            if mp[i] == protocol_name:
                mtarget.append(mtag_list[i])
        # Write to txt file
        with open(text_file_name2, 'a') as text_file:
            text_file.write(dataset + '\n')
            text_file.write(tag_name[1] + '\n\n')
            for k in range(len(mp)):
                text_file.write(mp[k] + '\n')
    except KeyError:
        print('No Songs found')

    f.close()
    return target, p, mtarget, mp


# ----------------------------------------------------------------------------------------------------------------------
# SPIKE STATISTICS


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


def psth(spike_times, n, bin_size, plot, return_values, separate_trials):
    # spike times must be seconds!
    # Compute histogram and calculate time dependent firing rate (binned PSTH)
    # n: number of trials
    # bin_size: bin size in seconds
    if separate_trials:
        spike_times = np.concatenate(spike_times)

    bins = int(np.max(spike_times) / bin_size)
    hist, bin_edges = np.histogram(spike_times, bins)
    # bin_width = bin_edges[1] - bin_edges[0]
    frate = hist / bin_size / n
    if plot:
        # Plot PSTH
        plt.plot(bin_edges[:-1], frate, 'k')

    if return_values:
        return frate, bin_edges
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


# ----------------------------------------------------------------------------------------------------------------------
# SPIKE TRAIN DISTANCE


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


def spike_e_pulses(spike_times, dt_factor, tau, duration):
    # tau in seconds
    # dt_factor: dt = tau/dt_factor
    # spike times in seconds
    # duration in seconds

    dt = tau / dt_factor

    # Remove spikes that are not of interest
    '''
    spike_times = spike_times[spike_times <= duration]
    if not spike_times.any():
        f = np.array([0])
        return f
    dur = np.max(spike_times) + 5 * tau
    '''

    dur = duration
    t = np.arange(0, dur, dt)
    f = np.zeros(len(t))

    for ti in spike_times:
        dummy = np.heaviside(t - ti, 0) * np.exp(-(t - ti) / tau)
        f = f + dummy

    return f


def vanrossum_distance(f, g, dt_factor, tau):

    # Make sure both trains have same length
    difference = abs(len(f) - len(g))
    if len(f) > len(g):
        g = np.append(g, np.zeros(difference))
    elif len(f) < len(g):
        f = np.append(f, np.zeros(difference))

    # Compute VanRossum Distance
    dt = tau/dt_factor
    d = np.sum((f - g) ** 2) * (dt / tau)
    # f_count = np.sum(f)* (dt / tau)
    # g_count = np.sum(g) * (dt / tau)
    return d


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


def vanrossum_matrix(dataset, tau, duration, dt_factor, boot_sample, save_fig):

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
    '''

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

    # Tags and Stimulus names
    connection = tagtostimulus(dataset)
    stimulus_tags = [''] * len(stims)
    for p in range(len(stims)):
        stimulus_tags[p] = connection[stims[p]]

    # Convert all Spike Trains to e-pulses
    trains = {}
    for k in range(len(stimulus_tags)):
        tr = [[]]*20
        for j in range(20):
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
        rand_ids = np.random.randint(20, size=call_count)
        for i in range(call_count):
            idx = np.arange(0, 20, 1)
            idx = np.delete(idx, rand_ids[k])
            templates.update({i: trains[stimulus_tags[i]][rand_ids[i]]})
            for q in range(len(idx)):
                probes.update({count: [trains[stimulus_tags[i]][idx[q]], i]})
                count += 1

        # Compute VanRossum Distance
        for pr in range(len(probes)):
            d = np.zeros(len(templates))
            for tmp in range(len(templates)):
                d[tmp] = vanrossum_distance(templates[tmp], probes[pr][0], dt_factor, tau)
            # What happens if there are two mins?
            template_match = np.where(d == np.min(d))[0][0]
            song_id = probes[pr][1]
            match_matrix[template_match, song_id] += 1

        mm.update({boot: match_matrix})

    mm_mean = sum(mm.values()) / len(mm)

    if save_fig:
        # Plot Matrix
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
        print('tau = ' + str(tau*1000) + ' ms' + ' T = ' + str(duration*1000) + ' ms done')

    return mm_mean


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# VECTOR STRENGTH AND BOOTSTRAPPING


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


# ----------------------------------------------------------------------------------------------------------------------
# MISC


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


def tagtostimulus(dataset):
    pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    tag_list = np.load(pathname + 'Calls_tag_list.npy')
    nix_file = '/media/brehm/Data/MasterMoth/mothdata/' + dataset + '/' + dataset + '.nix'
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    # mtags = {}
    connection = {}
    for k in range(len(tag_list)):
        # mtags.update({tag_list[k]: b.multi_tags[tag_list[k]]})
        mtag = b.multi_tags[tag_list[k]]
        sound_name = mtag.metadata.sections[0][2]
        connection.update({sound_name: tag_list[k]})

    f.close()
    return connection


def pytomat(dataset, protocol_name):
    # Load Voltage Traces
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    # tag_list = np.load(tag_list_path)
    for i in range(1, 90):
        volt = voltage['SingleStimulus-file-'+str(i)][0]
        scipy.io.savemat('/media/brehm/Data/volt_test/volt' + str(i) + '.mat', {'volt': volt})
    return 0


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


# ----------------------------------------------------------------------------------------------------------------------
# GAP PARADIGM


def gap_analysis(dataset, protocol_name):
    # Load Voltage Traces
    file_pathname = "/media/brehm/Data/MasterMoth/figs/" + dataset + "/DataFiles/"
    file_name = file_pathname + protocol_name + '_voltage.npy'
    tag_list_path = file_pathname + protocol_name + '_tag_list.npy'
    voltage = np.load(file_name).item()
    tag_list = np.load(tag_list_path)
    fs = 100*1000  # Sampling Rate of Ephys Recording

    # Set time range of single trials
    cut = np.arange(0, 16, 1.5)
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

