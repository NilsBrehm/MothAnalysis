from IPython import embed
import myfunctions as mf
import numpy as np
import nixio as nix
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
import thunderfish.peakdetection
from tqdm import tqdm



# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa'] # calls
# datasets = ['2017-11-01-aa'] # calls
# datasets = ['2017-12-05-aa']  # FI
# datasets = ['2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa', '2017-11-01-aa', '2017-11-16-aa']  # Carales FIs
datasets = ['2018-02-20-aa']

VIEWNIX = False
OVERVIEW = False
GetSession = False

FIFIELD = True

INTERVAL_MAS = False
INTERVAL_REC = False
GAP = False

SOUND = False
SOUND2 = False

PYTOMAT = False
CHECKPROTOCOLS = False
MAKEDIR = False

SELECT = True

# Select data
if SELECT:
    import csv
    p = os.path.join('..', 'overview.csv')
    with open(p, newline='') as f:
        datasets = []
        reader = csv.reader(f)
        for row in reader:
            if INTERVAL_MAS:
                if row[3] == 'True':  # this is MAS
                    datasets.append(row[0])
            if INTERVAL_REC:
                if row[2] == 'True':  # this is RectIntervals
                    datasets.append(row[0])
            if GAP:
                if row[1] == 'True':  # this is GAP
                    datasets.append(row[0])
            if FIFIELD:
                if row[4] == 'True' and row[6] == 'Creatonotos':  # this is FI
                    datasets.append(row[0])
    datasets = sorted(datasets)

# Get relative paths ===================================================================================================
# data_name = datasets[5]
# #  path_names = [data_name, data_files_path, figs_path, nix_path]
# path_names = mf.get_directories(data_name=data_name)

# Create Directory for Saving Data
if MAKEDIR:
    mf.make_directory(datasets[0])

# Session info
if GetSession:
    # Get all Recordings in "/mothdata/"
    file_list = os.listdir(path_names[3])
    file_list = sorted([st for st in file_list if '20' in st])
    mf.get_session_metadata(file_list)

if VIEWNIX:
    # Get all Recordings in "/mothdata/"
    file_list = os.listdir(path_names[3])
    file_list = sorted([st for st in file_list if '20' in st])

    # Get all stimulus information
    for i in range(len(file_list)):
        data_set = file_list[i]

        mf.make_directory(data_set)

        # List all tags and multi tags
        r = mf.view_nix(data_set)
        if r == 1:
            continue

        # Closer look at tags
        target, stim_list, _, songs = mf.list_protocols(data_set, protocol_name='Gap',
                                                        tag_name=['SingleStimulus_', 'SingleStimulus-file-'])
    print('All stimulus infos saved')

# FIField
if FIFIELD:
    species = 'Carales'

    if species is 'Estigmene':
        # Estigmene:
        datasets = ['2017-11-25-aa', '2017-11-27-aa']  # 20 ms
        # datasets = ['2017-10-26-aa', '2017-11-25-aa', '2017-11-27-aa', '2017-12-05-aa']
    elif species is 'Carales':
        # Carales:
        # datasets = ['2017-11-01-aa', '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']  # 20 ms
        datasets = ['2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac']  # 50 ms
        # datasets = ['2017-10-23-ah', '2017-10-30-aa', '2017-10-31-aa', '2017-10-31-ac', '2017-11-01-aa',
        #             '2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa']

    spike_detection = True
    show_detection = True
    save_spikes = False
    collect_volt = False

    print('Starting FIField Data Gathering')
    if collect_volt:
        for k in tqdm(range(len(datasets)), desc='Data Sets'):
            data_set_number = k
            data_name = datasets[data_set_number]
            print(str(data_set_number+1) + ' of ' + str(len(datasets)))
            print(data_name)
            path_names = mf.get_directories(data_name=data_name)
            mf.fifield_voltage2(path_names, rec_dur_factor=2)
    if spike_detection:
        valley = [False] * len(datasets)
        # valley[0] = True
        # valley[1] = True
        for k in tqdm(range(len(datasets)), desc='Data Sets'):
            data_set_number = k
            data_name = datasets[data_set_number]
            print(str(data_set_number+1) + ' of ' + str(len(datasets)))
            print(data_name)
            if valley[k]:
                print('Valley Mode selected')
            path_names = mf.get_directories(data_name=data_name)
            mf.fifield_spike_detection(path_names, th_factor=3, th_window=None, mph_percent=2, filter_on=True,
                                       valley=valley[k], min_th=50, save_data=save_spikes, show=show_detection)

# Intervals: MothASongs
if INTERVAL_MAS:
    print('Starting Moth Intervals Data Gathering')
    data_name = datasets[-18]
    print(data_name)
    #  path_names = [data_name, data_files_path, figs_path, nix_path]
    path_names = mf.get_directories(data_name=data_name)

    mf.get_moth_intervals_data(path_names, save_data=True)



# Rect Intervals
if INTERVAL_REC:
    protocol_name = 'PulseIntervalsRect'
    target, p, mtarget, mp = mf.list_protocols(path_names, protocol_name,
                                               tag_name=['SingleStimulus_', 'SingleStimulus-file-'], save_txt=False)
    voltage, tag_list = mf.get_voltage_trace_gap(path_names, target, protocol_name, multi_tag=False,
                                                     search_for_tags=False, save_data=True)

# Sound Recording Stimuli
if SOUND:
    # Create Directory for Saving Data
    pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/naturalmothcalls/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/batcalls/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/batcalls/noisereduced/"
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory

    recordings = 59
    for i in range(recordings):
        mf.get_soundfilestimuli_data(datasets, 'SingleStimulus-file-' + str(i+1), False)
        # input("Press Enter to continue...")
    print('Files saved')


if SOUND2:
    # mf.get_metadata(datasets[0], 'MothASongs-moth_song-damped_oscillation*', 'Intervals')
    # mf.get_metadata(datasets[0], 'SingleStimulus-file-', 'Calls')
    mf.get_voltage_trace(path_names, 'SingleStimulus-file-', 'Calls', multi_tag=True, search_for_tags=True)

if PYTOMAT:
    mf.pytomat(path_names, 'Calls')


if GAP:
    gaps, p, _, _ = mf.list_protocols(path_names, protocol_name='Gap',
                                      tag_name=['SingleStimulus_', 'SingleStimulus-file-'], save_txt=False)
    mf.get_voltage_trace_gap(path_names, gaps, 'Gap', multi_tag=False, search_for_tags=False, save_data=True)


if OVERVIEW:
    mf.overview_recordings(look_for_mtags=False)

print('Overall Data Gathering done')
