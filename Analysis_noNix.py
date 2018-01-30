from IPython import embed
import myfunctions as mf
import numpy as np
import os
import matplotlib.pyplot as plt

# Only a test comment
# Data File Name
# datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']
datasets = ['2017-11-02-ad']

"""
data_name = datasets[0]
pathname = "figs/" + data_name + "/"
filename = '/media/brehm/Data/mothdata/' + data_name + '/' + 'fifield.dat'
"""

# fifield.dat:
# threshold                             rate                   saturation
# f_c     I_th    s.d.    slope  s.d.   I_r     s.d.    r      I_max   s.d.    f_max  s.d.
# kHz     dB SPL  dB SPL  Hz/dB  Hz/dB  dB SPL  dB SPL  Hz     dB SPL  dB SPL  Hz     Hz
#      1       2       3      4      5       6       7      8       9      10     11     12

for i in range(len(datasets)):
    pathname = "/media/brehm/Data/MasterMoth/figs/" + datasets[i] + "/"
    filename = '/media/brehm/Data/MasterMoth/mothdata/' + datasets[i] + '/' + 'fifield.dat'
    # Create Directory for Saving Data
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory
    data_fifield = np.loadtxt(filename)
    n = 10
    idx = np.argsort(data_fifield[:, 0])  # Sort after freqs
    plt.figure()
    data = data_fifield
    plt.errorbar(data_fifield[idx, 0], data_fifield[idx, 1], yerr=data_fifield[idx, 2]/np.sqrt(n), fmt='k-o')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Threshold [dB SPL]')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.text(80, 70, 'repeats = 10\n 1 animal')
    # Save Plot to HDD
    figname = pathname + "FIField_relacs.png"
    fig = plt.gcf()
    fig.set_size_inches(16, 12)
    fig.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close(fig)

print('All plots saved')

