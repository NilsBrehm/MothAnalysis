from IPython import embed
import myfunctions as mf
import numpy as np
import nixio as nix
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
datasets = ['2017-11-17-aa']

# Rect Intervals
mf.get_rect_intervals_data(datasets)

embed()

# FIField
print('Starting FIField Data Gathering')
mf.get_fifield_data(datasets)

# Intervals: MothASongs
print('Starting Moth Intervals Data Gathering')
mf.get_moth_intervals_data(datasets)

# Sound Recording Stimuli
# mf.get_soundfilestimuli_data(datasets, 'SingleStimulus-file-5')

# Pulse Intervals

print('Overall Data Gathering done')
