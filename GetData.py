from IPython import embed
import myfunctions as mf
import numpy as np
import nixio as nix
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
import thunderfish.peakdetection

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa'] # calls
# datasets = ['2017-11-01-aa'] # calls
# datasets = ['2017-12-05-aa']  # FI
datasets = ['2018-02-20-aa']
# datasets = ['2017-11-02-aa', '2017-11-02-ad', '2017-11-03-aa', '2017-11-01-aa', '2017-11-16-aa']  # Carales FIs

VIEWNIX = False
GetSession = False

FIFIELD = False

INTERVAL_MAS = False
INTERVAL_REC = False
GAP = True

SOUND = False
SOUND2 = False

PYTOMAT = False
CHECKPROTOCOLS = False
MAKEDIR = False



# Create Directory for Saving Data
if MAKEDIR:
    mf.make_directory(datasets[0])

# Session info
if GetSession:
    # Get all Recordings in "/mothdata/"
    file_list = os.listdir('/media/brehm/Data/MasterMoth/mothdata/')
    file_list = sorted([st for st in file_list if '20' in st])
    mf.get_session_metadata(file_list)

if VIEWNIX:
    # Get all Recordings in "/mothdata/"
    file_list = os.listdir('/media/brehm/Data/MasterMoth/mothdata/')
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
    print('Starting FIField Data Gathering')
    # mf.fifield_voltage2(datasets[0], 'FIField-sine_wave-1')
    mf.fifield_spike_detection(datasets[0], th_factor=2, th_window=None, mph_percent=10, filter_on=True, valley=False,
                              min_th=50, save_data=False)

# Intervals: MothASongs
if INTERVAL_MAS:
    print('Starting Moth Intervals Data Gathering')
    mf.get_moth_intervals_data(datasets)

# Rect Intervals
if INTERVAL_REC:
    mf.get_rect_intervals_data(datasets)

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
    mf.get_voltage_trace(datasets[0], 'SingleStimulus-file-', 'Calls', multi_tag=True, search_for_tags=True)

if PYTOMAT:
    mf.pytomat(datasets[0], 'Calls')

if CHECKPROTOCOLS:
    gaps, p = mf.list_protocols(datasets[0], 'Gap')
    # voltage, tag_list = mf.get_voltage_trace(datasets[0], gaps, 'Gap', multi_tag=False ,search_for_tags=False)
    mf.gap_analysis(datasets[0], 'Gap')
    embed()

if GAP:
    # gaps, p, _, _ = mf.list_protocols(datasets[0], protocol_name='Gap',
    #                                   tag_name=['SingleStimulus_', 'SingleStimulus-file-'])
    # mf.get_voltage_trace(datasets[0], gaps, 'Gap', multi_tag=False, search_for_tags=False)
    #
    # # Cut out single trials from recording
    # mf.gap_analysis(datasets[0], 'Gap')
    #
    mf.tagtostimulus_gap(datasets[0])



print('Overall Data Gathering done')
