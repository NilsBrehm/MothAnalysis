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
datasets = ['2018-01-26-ab']

GetSession = True
FIFIELD = False
INTERVAL_MAS = False
INTERVAL_REC = False
SOUND = True

# Create Directory for Saving Data
pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/"
directory = os.path.dirname(pathname)
if not os.path.isdir(directory):
    os.mkdir(directory)  # Make Directory

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
    pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[0] + "/mothsongs/"
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

    recordings = 21
    for i in range(recordings):
        mf.get_soundfilestimuli_data(datasets, 'SingleStimulus-file-' + str(i+1), False)
        # input("Press Enter to continue...")
    print('Files saved')

# Session info
if GetSession:
    mf.get_session_metadata(datasets)

print('Overall Data Gathering done')
