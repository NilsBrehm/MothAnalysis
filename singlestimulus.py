import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
import time
import myfunctions
from IPython import embed

# ----------------------------------------
# Data File Name
data_name = '2017-10-06-aa'
data_file = '/home/brehm/data/' + data_name + '/' + data_name + '.nix'

# ----------------------------------------
# Open the nix file
f = nix.File.open(data_file, nix.FileMode.ReadOnly)
b = f.blocks[0]

""" Get tagged spikes and make raster plot, instantaneous firing rate and stimulus"""
########################################################################################################################
# Set parameters:
protocol_count = 10       # Number of executed protocols
sample_size_limit = 20      # Number of repeats in one protocol
samplingrate = 20 * 1000   # sampling rate of recording
# ----------------------------------------------------------------------------------------------------------------------

# Loop through all stimulus protocols
for j in range(protocol_count):
    j += 1

    # ----------------------------------------
    # Get desired tags
    tagname = "MothASongs-moth_song-damped_oscillation*10-" + str(j)
    tag = b.multi_tags[tagname]  # multi tag

    # ----------------------------------------
    # Get Features
    onsettag = tagname + " onset times"
    stimstart = b.data_arrays[onsettag]
    amplitude = tag.features[4].data
    samplesize = list(amplitude.shape)

    # ----------------------------------------
    # Reduce samplesize
    if samplesize[0] < 2:
        print("Not enough samples:")
        print(tagname + " was excluded from analysis")
        continue
    if samplesize[0] > sample_size_limit:
        samplesize[0] = sample_size_limit

    # ----------------------------------------
    # Get Metadata
    meta = tag.metadata.sections[0]
    meta_stim = meta["MothASongs-moth_song-damped_oscillation*10-1"]
    duration = meta_stim["Duration"]
    dutycycle = meta_stim["DutyCycle"]
    frequency = meta_stim["Frequency"]
    meta_carrier = meta["stimulus_sine_wave"]
    carrierfreq = meta_carrier["Frequency"]

    # ----------------------------------------
    # Now get spike times for tag
    sp = {}
    y = {}
    # rate = {}
    frate = np.zeros((samplesize[0], duration * samplingrate))
    volt = np.zeros((samplesize[0], 996))
    # volt1 = {}
    for i in range(samplesize[0]):  # Loop through all repeats in this protocol
        # Get Spike Times
        spikes = myfunctions.get_spiketimes(tag, 1, i)
        spikes = spikes - stimstart[i]  # Align times to stimulus onset at 0 sec
        sp.update({i: spikes})
        y.update({i: np.ones(spikes.shape) * i})

        # Calculate instantaneous firing rate
        _, r = myfunctions.inst_firing_rate(spikes, duration, 1 / samplingrate)
        frate[i] = r
        # rate.update({i : r})

        # Get voltage
        volt[i] = myfunctions.get_voltage(tag, 1, i)[:]
        # volt1.update({i: myfunctions.get_voltage(tag, 1, i)[:]})

    t, _ = myfunctions.inst_firing_rate(spikes, duration, 1 / samplingrate)
    mean_rate = np.mean(frate, axis=0)
    mean_volt = np.mean(volt, axis=0)
    mean_volt_time = np.linspace(0, duration, mean_volt.shape[0])

    # ----------------------------------------
    # Recreate Stimulus settings: time in msec!
    N = duration * 1000  # sample count = stimulus length
    freq = frequency
    dutycycle = dutycycle * 100
    period = 20
    pulseduration = 10
    stimulus, p, pd = (myfunctions.square_wave(N, freq, dutycycle, period, pulseduration))
    stimulus = stimulus * amplitude[0]  # sets stimulus to correct amplitude
    stimulus_timeax = np.linspace(0, N / 1000, N)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Make Subplot
    plt.figure(j)
    plt.subplot(4, 1, 1)
    for k in range(samplesize[0]):  # Loop through all trials in this protocol
        plt.plot(sp[k], y[k], "|", markersize=10, markeredgewidth=2, color="black")
    plt.title("Spike Trains")
    plt.ylabel('Trial #')
    plt.xlim(0, duration)
    plt.ylim(0, samplesize[0])

    plt.subplot(4, 1, 2)
    plt.plot(t, mean_rate, color="black")
    plt.title("Mean Instantaneous Firing Rate")
    plt.ylabel("Spikes/sec")

    plt.subplot(4, 1, 3)
    plt.plot(mean_volt_time, mean_volt, color='black')
    plt.ylabel("Mean Voltage [uV]")

    plt.subplot(4, 1, 4)
    stim_params = "period = " + str(p) + " ms and pulseduration =  " + str(pd) + " ms"
    stimplot = plt.plot(stimulus_timeax, stimulus, linewidth=4, color="black")
    ti = "Stimulus with carrier frequency " + str(carrierfreq / 1000) + " kHz"
    plt.title(stim_params)
    plt.suptitle(ti)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude [db SPL]')
    plt.xlim(0, duration)
    embed()
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Write figures to HDD
    # plt.show()
    figname = "figs/" + data_name + "/SingleStimulus/" + str(j) + ".png"
    fig = plt.gcf()
    fig.set_size_inches(16, 12)
    fig.savefig(figname, bbox_inches='tight', dpi=100)
    plt.close(fig)
    ladebalken = "total:" + str(round((j / protocol_count) * 100)) + " % completed"
    print(tagname)
    print("samplesize: " + str(samplesize[0]))
    print(ladebalken)
    print("")

f.close()
