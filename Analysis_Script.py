from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
# datasets = ['2017-11-17-aa', '2017-11-16-aa', '2017-11-14-aa']
datasets = ['2017-11-17-aa']

peak_params = {'mph': 80, 'mpd': 50, 'valley': False, 'show': False}

# Test Comment 2
# Rect Intervals
spikes = mf.rect_intervals_spike_detection(datasets, peak_params)
mf.rect_intervals_cut_trials(datasets)

embed()
# Analyse Intervals MothASongs data stored on HDD
mf.moth_intervals_spike_detection(datasets, peak_params, -1)
mf.moth_intervals_analysis(datasets)

# Analyse FIField data stored on HDD
spike_threshold = 4
mf.fifield_analysis(datasets, spike_threshold, peak_params)

print('Analysis done!')
