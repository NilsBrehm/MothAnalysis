import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
from IPython import embed
import time
import os
import myfunctions as mf

start_time = time.time()
# Data File Name
data_name = '2017-11-02-ad'
data_file = '/media/brehm/Data/mothdata/' + data_name + '/' + data_name + '.nix'

# Open the nix file
f = nix.File.open(data_file, nix.FileMode.ReadOnly)
b = f.blocks[0]

# Get tags
mtag = b.multi_tags['FIField-sine_wave-1']

# Get Meta Data
meta = mtag.metadata.sections[0]
duration = meta["Duration"]
frequency = mtag.features[5].data[:]
amplitude = mtag.features[4].data[:]
final_data = np.zeros((len(frequency), 3))

# Create Directory for Saving Plots
pathname = "figs/" + data_name + "/"
directory = os.path.dirname(pathname)
if not os.path.isdir(directory):
    os.mkdir(directory)  # Make Directory

# Set Threshold for FIField
spikethreshold = 3  # in spike count

# Get data for all frequencies
freq = np.unique(frequency)  # All used frequencies in experiment
qq = 0
ficurve = {}
db_threshold = np.zeros((len(freq), 2))
timeneeded = np.zeros((len(freq)))
for q in freq[5:7]:
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
        # percent = np.round(k / len(d), 3)
        # print(percent*100)

    # Now average all spike counts per amplitude
    amps = np.unique(data[:, 1])  # All used amplitudes in experiment
    k = 0
    average = np.zeros((len(amps), 4))
    for i in amps:
        ids = np.where(data == i)[0]
        average[k, 0] = i                     # Amplitude
        average[k, 1] = np.mean(data[ids, 2])  # Mean spike number per presentation
        average[k, 2] = np.std(data[ids, 2])   # STD spike number per presentation
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
    # dname = pathname + 'FICurve_' + str(int(q/1000)) + '.npy'
    # np.save(dname, average)

    # Now plot FI  Curve and save it to HDD
    # mf.plot_ficurve(average, q, pathname)

    # Estimate Time Remaining for Analysis
    percent = np.round((qq+1) / len(freq), 3)
    print('--- %s %% done ---' % (percent*100))
    intertime2 = time.time()
    timeneeded[qq] = np.round((intertime2 - intertime1)/60, 2)
    if qq > 0:
        timeremaining = np.round(np.mean(timeneeded[0:qq]), 2) * (len(freq)-qq)
    else:
        timeremaining = np.round(np.mean(timeneeded[qq]), 2) * (len(freq) - qq)
    print('--- Time remaining: %s minutes ---' % timeremaining)
    qq += 1

# Plot FI Field and save it to HDD
# mf.plot_fifield(db_threshold, pathname)

# Save Data to HDD
dname = pathname + 'FICurve.npy'
dname2 = pathname + 'FIField.npy'
np.save(dname2, db_threshold)
np.save(dname, ficurve)


# PLOTTING SECTION
# Load data from HDD
ficurves, fifield = mf.loading_fifield_data(pathname)

# Plot FieField and FICurves and save to HDD
freq = [30000, 35000]
mf.plotting_fifield_data(pathname, freq, fifield, ficurves)

f.close()  # Close Nix File
print('All Done!')
print("--- Analysis took %s minutes ---" % np.round((time.time() - start_time)/60))

