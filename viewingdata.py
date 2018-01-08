import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
import time
import myfunctions
from IPython import embed
from collections import Counter as cct

start_time = time.time()
# Data File Name
data_name = '2017-10-26-aa'
data_file = '/home/brehm/data/' + data_name + '/' + data_name + '.nix'

""" Get tagged spikes and plot firing rate vs amplitude and plot FI Field """
########################################################################################################################
# Set parameters:
limit = 5  # Spikes/Stimulus
number_of_protocols = 1
# ----------------------------------------------------------------------------------------------------------------------
# Open the nix file
f = nix.File.open(data_file, nix.FileMode.ReadOnly)
b = f.blocks[0]

# Choose desired tag
maxi = number_of_protocols+1
threshold = np.zeros((maxi, 2))
for nr in range(1, maxi):
    tagname = "FIField-sine_wave-" + str(nr)
    print(tagname)
    m = b.multi_tags[tagname]
    spike_count = myfunctions.get_spike_count(m)  # Takes 5 seconds....
    spike_count = np.delete(spike_count, 0, 0)  # delete buggy first 0 entry
    c = cct(spike_count[:, 1])  # Count how many entries there are pro dB value
    counter = len(c)
    mean_spike_count = np.zeros((counter, 2))
    i = 0
    for k in c.keys():
        id1 = spike_count[:, 1] == k
        mean_spike_count[i, 0] = np.mean(spike_count[id1, 0])
        mean_spike_count[i, 1] = k
        i += 1
    mean_spike_count = np.sort(mean_spike_count, axis=0)
    # Get MetaData
    duration, frequency, amplitude = myfunctions.get_metadata(m)
    ti = str(frequency) + " Hz"

    # Determine threshold: dB SPL that elicit min. x spikes/stimulus
    # threshold = [frequency, dB SPL]
    threshold[nr-1, 0] = frequency
    if np.max(mean_spike_count[:, 0]) < limit:
        print("Spikes/Stimulus under threshold limit: " + str(limit))
        print(tagname+" will be excluded from analysis")
        print("done: ", round((nr / (maxi - 1)) * 100), " %")
        print("")
        continue
    threshold[nr-1, 1] = np.min(mean_spike_count[:, 1][mean_spike_count[:, 0] >= limit])

    # Plot FI Curve for this protocol
    fig = plt.figure(nr)
    plt.plot(mean_spike_count[:, 1], mean_spike_count[:, 0], "-o", color="black")
    u = np.ones((mean_spike_count[:, 1].shape[0]))*limit
    plt.plot(mean_spike_count[:, 1], u, "--", color="red")
    plt.xlabel("Amplitude [db SPL]")
    plt.ylabel("Spikes/Stimulus")
    plt.title(ti)
    plt.xticks(np.arange(20, 120, 10.0))
    plt.yticks(np.arange(0, 100, 5.0))
    figname = "figs/"+data_name+"/amp/" + ti + ".png"
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)
    print("done: ", round((nr/(maxi-1))*100), " %")

# Now sort thresholds with increasing frequency
threshold = threshold[threshold[:, 0].argsort()]

# Find zero entries and remove them
threshold = np.delete(threshold, np.where(threshold[:, 1] == 0), 0)

# Plot FIField and write to HDD
plt.plot(threshold[:, 0], threshold[:, 1], '-o', color='black')
plt.title("FI-Field: Threshold= " + str(limit) + " Spikes/Stimulus")
plt.xlabel("Frequency in Hz")
plt.ylabel("Threshold Amplitude in db SPL")
plt.xticks(np.arange(np.min(threshold[:, 0]), np.max(threshold[:, 0])+10001, 5000))
plt.yticks(np.arange(0, np.max(threshold[:, 1])+10, 10))
figname1 = "figs/"+data_name+"/FIField/" + "FIField.png"
fig1 = plt.gcf()
fig1.set_size_inches(16, 12)
fig1.savefig(figname1, bbox_inches='tight')
plt.show()
f.close()
print("--- %s seconds ---" % (time.time() - start_time))
