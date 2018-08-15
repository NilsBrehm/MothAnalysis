from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
datasets = ['2017-11-16-aa']

for all_datasets in range(len(datasets)):
    nix_name = datasets[all_datasets]
    pathname = "figs/" + nix_name + "/"
    filename = pathname + 'FIField_voltage.npy'


    # Load data from HDD
    fifield_volt = np.load(filename).item()  # Load dictionary data
    freqs = np.load(pathname + 'frequencies.npy')
    amps = np.load(pathname + 'amplitudes.npy')
    amps_uni = np.unique(np.round(amps))
    freqs_uni = np.unique(freqs)/1000

    # Find Spikes
    fi = {}
    for f in range(len(freqs_uni)):  # Loop through all different frequencies
        dbSPL = {}
        for a in fifield_volt[freqs_uni[f]].keys():  # Only loop through amps that exist
            repeats = len(fifield_volt[freqs_uni[f]][a])-1
            spike_count = np.zeros(repeats)
            for trial in range(repeats):
                x = fifield_volt[freqs_uni[f]][a][trial]
                spike_count[trial] = len(mf.detect_peaks(x, mph=30, mpd=50, threshold=0, edge='rising', kpsh=False, valley=True, show=False, ax=None))
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
        spike_threshold = 8
        th = mean_spike_count >= spike_threshold
        dbSPL_threshold[f, 0] = freqs_uni[f]
        if th.any():
            dbSPL_threshold[f, 1] = amplitude_sorted[np.min(np.where(th))]
        else:
            dbSPL_threshold[f, 1] = 'NaN'

        # Save FIField data to HDD
        dname1 = pathname + 'FICurve_' + str(freqs_uni[f])
        np.savez(dname1, amplitude_sorted=amplitude_sorted, mean_spike_count=mean_spike_count,
                 spike_threshold=spike_threshold, dbSPL_threshold=dbSPL_threshold, freq=freqs_uni[f])
        # Plot FICurve for this frequency
        mf.plot_ficurve(amplitude_sorted, mean_spike_count, std_spike_count, freqs_uni[f], spike_threshold, pathname, savefig=True)
        # Estimate Progress
        percent = np.round(f/len(freqs_uni), 2)
        print('--- %s %% done ---' % (percent * 100))

    # Save FIField data to HDD
    dname = pathname + 'FIField_plotdata.npy'
    np.save(dname, dbSPL_threshold)

    dname2 = pathname + 'FIField_FISpikes.npy'
    np.save(dname2, fi)

    # Plot FIField and save it to HDD
    mf.plot_fifield(dbSPL_threshold, pathname, savefig=True)
    print('Analysis finished: %s' % datasets[all_datasets])

    # Progress Bar
    percent = np.round((all_datasets + 1) / len(datasets), 2)
    print('-- Analysis total: %s %%  --' % (percent * 100))
