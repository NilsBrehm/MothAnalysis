from IPython import embed
import myfunctions as mf
import numpy as np
import nixio as nix
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
# datasets = ['2018-02-09-aa']
datasets = ['2018-02-20-aa']

GetSession = False
FIFIELD = False
INTERVAL_MAS = False
INTERVAL_REC = False
SOUND = False
SOUND2 = False
PYTOMAT = False
CHECKPROTOCOLS = True
TEST = False

# Create Directory for Saving Data
mf.make_directory(datasets[0])

# FIField
if FIFIELD:
    print('Starting FIField Data Gathering')
    mf.get_fifield_data(datasets)

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

# Session info
if GetSession:
    mf.get_session_metadata(datasets)

if SOUND2:

    # mf.get_metadata(datasets[0], 'MothASongs-moth_song-damped_oscillation*', 'Intervals')
    # mf.get_metadata(datasets[0], 'SingleStimulus-file-', 'Calls')
    mf.get_voltage_trace(datasets[0], 'SingleStimulus-file-', 'Calls', multi_tag=True ,search_for_tags=True)

if PYTOMAT:
    mf.pytomat(datasets[0], 'Calls')


if CHECKPROTOCOLS:
    gaps, p = mf.list_protocols(datasets[0], 'Gap')
    # voltage, tag_list = mf.get_voltage_trace(datasets[0], gaps, 'Gap', multi_tag=False ,search_for_tags=False)
    # mf.gap_analysis(datasets[0], 'Gap')
    embed()

if TEST:
    v = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Gap_voltage.npy').item()
    # x = v['SingleStimulus_112']

    voltage = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-09-aa/DataFiles/Calls_voltage.npy').item()
    x = voltage['SingleStimulus-file-2'][0]
    locs = mf.indexes(x, th_factor=2, min_dist=50)
    embed()
    exit()
    mf.peak_seek(x, 100, 100)

print('Overall Data Gathering done')
