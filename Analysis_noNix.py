from IPython import embed
import myfunctions as mf
import numpy as np
import os
import matplotlib.pyplot as plt

# Data File Name
datasets = ['2017-11-03-aa', '2017-11-02-ad', '2017-11-02-ac', '2017-11-02-ab', '2017-11-02-aa', '2017-11-01-aa']

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
    pathname = "figs/" + datasets[i] + "/"
    filename = '/media/brehm/Data/mothdata/' + datasets[i] + '/' + 'fifield.dat'
    # Create Directory for Saving Data
    directory = os.path.dirname(pathname)
    if not os.path.isdir(directory):
        os.mkdir(directory)  # Make Directory
    data_fifield = np.loadtxt(filename)
    plt.figure()
    plt.errorbar(data_fifield[:, 0], data_fifield[:, 1], yerr=data_fifield[:, 2], fmt='k-o')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Threshold [dB SPL]')
    plt.xticks(np.linspace(10, 110, 11))
    plt.text(60, 60, 'repeats = 5\n 1 animal')
    # Save Plot to HDD
    figname = pathname + "FIField.png"
    fig = plt.gcf()
    fig.set_size_inches(16, 12)
    fig.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close(fig)

print('All plots saved')

