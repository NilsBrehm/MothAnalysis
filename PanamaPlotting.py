import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import scipy.io.wavfile as wav
import scipy as scipy
import matplotlib
import scipy.io.wavfile as wav
import seaborn as sns
from scipy import signal as sg


def plot_settings():
    # Font:
    # matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 8

    # Ticks:
    matplotlib.rcParams['xtick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.size'] = 4
    matplotlib.rcParams['xtick.major.size'] = 4

    # Title Size:
    matplotlib.rcParams['axes.titlesize'] = 10

    # Axes Label Size:
    matplotlib.rcParams['axes.labelsize'] = 8

    # Axes Line Width:
    matplotlib.rcParams['axes.linewidth'] = 1

    # Tick Label Size:
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6

    # Line Width:
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['lines.color'] = 'k'

    # Marker Size:
    matplotlib.rcParams['lines.markersize'] = 2

    # Error Bars:
    matplotlib.rcParams['errorbar.capsize'] = 0

    # Legend Font Size:
    matplotlib.rcParams['legend.fontsize'] = 5

    return matplotlib.rcParams


def voltage_trace_filter(voltage, cutoff, order, ftype, filter_on=True):
    if filter_on:
        b, a = sg.butter(order, cutoff, btype=ftype, analog=False)
        y = sg.filtfilt(b, a, voltage)
    else:
        y = voltage
    return y


# Color Map

C = [[0, 0, 0],
[0, 0, 0.0476190485060215],
[0, 0, 0.0952380970120430],
[0, 0, 0.142857149243355],
[0, 0, 0.190476194024086],
[0, 0, 0.238095238804817],
[0, 0, 0.285714298486710],
[0, 0, 0.333333343267441],
[0, 0, 0.380952388048172],
[0, 0, 0.428571432828903],
[0, 0, 0.476190477609634],
[0, 0, 0.523809552192688],
[0, 0, 0.571428596973419],
[0, 0, 0.619047641754150],
[0, 0, 0.666666686534882],
[0, 0, 0.714285731315613],
[0, 0, 0.761904776096344],
[0, 0, 0.809523820877075],
[0, 0, 0.857142865657806],
[0, 0, 0.904761910438538],
[0, 0, 0.952380955219269],
[0, 0, 1],
[0, 0.0833333358168602, 0.916666686534882],
[0, 0.166666671633720, 0.833333313465118],
[0, 0.250000000000000, 0.750000000000000],
[0, 0.333333343267441, 0.666666686534882],
[0, 0.416666656732559, 0.583333313465118],
[0, 0.500000000000000, 0.500000000000000],
[0, 0.541666686534882, 0.458333343267441],
[0, 0.583333313465118, 0.416666656732559],
[0, 0.625000000000000, 0.375000000000000],
[0, 0.666666686534882, 0.333333343267441],
[0, 0.708333313465118, 0.291666656732559],
[0, 0.750000000000000, 0.250000000000000],
[0, 0.791666686534882, 0.208333328366280],
[0, 0.833333313465118, 0.166666671633720],
[0, 0.875000000000000, 0.125000000000000],
[0, 0.916666686534882, 0.0833333358168602],
[0, 0.958333313465118, 0.0416666679084301],
[0, 1, 0],
[0.125000000000000, 1, 0],
[0.250000000000000, 1, 0],
[0.375000000000000, 1, 0],
[0.500000000000000, 1, 0],
[0.625000000000000, 1, 0],
[0.750000000000000, 1, 0],
[0.875000000000000, 1, 0],
[1, 1, 0],
[0.990686297416687, 0.958088219165802, 0.00637254910543561],
[0.981372535228729, 0.916176497936249, 0.0127450982108712],
[0.972058832645416, 0.874264717102051, 0.0191176477819681],
[0.962745070457459, 0.832352936267853, 0.0254901964217424],
[0.953431367874146, 0.790441155433655, 0.0318627469241619],
[0.944117665290833, 0.748529434204102, 0.0382352955639362],
[0.934803903102875, 0.706617653369904, 0.0446078442037106],
[0.925490200519562, 0.664705872535706, 0.0509803928434849],
[0.916176497936249, 0.622794151306152, 0.0573529414832592],
[0.906862735748291, 0.580882370471954, 0.0637254938483238],
[0.897549033164978, 0.538970589637756, 0.0700980424880981],
[0.888235330581665, 0.497058838605881, 0.0764705911278725],
[0.878921568393707, 0.455147057771683, 0.0828431397676468],
[0.869607865810394, 0.413235306739807, 0.0892156884074211],
[0.860294103622437, 0.371323525905609, 0.0955882370471954],
[0.850980401039124, 0.329411774873734, 0.101960785686970]]
cm_selena = matplotlib.colors.ListedColormap(C)

# Data
# Carales_astur/PK1285/Pk12850017/call_nr_2
# Melese_incertus/Pk1299/Pk12990020/call_nr_3
# PP267_A80530003_50/call_nr_1
# /GL005/BCI1349
data_path = '/media/nils/Data/Panama/Recordings/'
# species = 'Melese_incertus'
# species = 'Carales_astur'
species = 'GL005'
# animal = 'PK1285'
animal = 'BCI1349'
# recording_nr = 'Pk12850017'
recording_nr = 'BCI13490001_m_11a11p'
call_nr = 1

file_name = data_path + species + '/' + animal + '/' + recording_nr + '/call_nr_' + str(call_nr) + '/matrix_analysis.mat'
audio_name = data_path + species + '/' + animal + '/' + recording_nr + '/call_nr_' + str(call_nr) + '.wav'
mat_file = scipy.io.loadmat(file_name)
fs, x = wav.read(audio_name)

# High Pass Filter Audio
nyqst = 0.5 * fs
lowcut = 2000
highcut = 100000
low = lowcut / nyqst
high = highcut / nyqst
audio_call = voltage_trace_filter(x, low, ftype='high', order=2, filter_on=True)

# Store Data
AvsA = mat_file['MaxCorr_AA']
PvsP = mat_file['MaxCorr_PP']
AvsP = mat_file['MaxCorr_AP']

matrix_data = {0: mat_file['MaxCorr_AA'], 1: mat_file['MaxCorr_PP'], 2: mat_file['MaxCorr_AP']}

# Pulses: 0=Active, 1=Passive
pulses = {0: mat_file['pulses'][0][0][0], 1: mat_file['pulses'][0][0][1]}

# Spectrogram: FFT > Nx !
Nx = 256
FFT = 512
nover = Nx-5
w = scipy.signal.get_window('hann', Nx, fftbins=True)
f, t, Sxx = scipy.signal.spectrogram(audio_call, fs, window=w, nperseg=None, noverlap=nover, nfft=FFT,
                                     detrend='constant', return_onesided=True, scaling='density', axis=-1,
                                     mode='psd')

db = 20*np.log10(Sxx/np.max(Sxx))

# Histogram of db values
h, b = np.histogram(db, 100)
bins = b[:-1]
idx = h == np.max(h)
min_db = np.round(bins[idx][0])

# plt.figure(figsize=(10, 2.5))
# plt.pcolormesh(t, f, db, cmap=cm_selena, vmin=-100, vmax=0, shading='gouraud')
# plt.colorbar()
# plt.ylim(0, 80000)

plot_settings()
ax = [[]] * 6
grid = matplotlib.gridspec.GridSpec(nrows=133, ncols=56)
fig = plt.figure(figsize=(5.9, 3.9))
n_active = pulses[0].shape[1]
n_passive = pulses[1].shape[1]
n = np.max([n_active, n_passive])

# Grid Spectrogram
ax[0] = plt.subplot(grid[0:20, 0:20])
ax_cb_spec = plt.subplot(grid[0:20, 21:22])

# Grid Matrix
ax[1] = plt.subplot(grid[0:30, 35:45])
ax[2] = plt.subplot(grid[50:80, 35:45])
ax[3] = plt.subplot(grid[100:130, 35:45])

# Grid Single Pulses
ax_a = grid[45:130, 0:10]
ax_p = grid[45:130, 12:22]

# Grid Color Bar Matrix
ax_cb = plt.subplot(grid[0:132, 46:47])

# Subplot caps
subfig_caps = 12
label_x_pos = -0.6
label_y_pos = 1.1
subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

# Single Pulses
x_lim = 0.2
t_a = np.linspace(0, len(pulses[0][:, n-1])/fs, len(pulses[1][:, n-1])) * 1000
grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(n, 1, subplot_spec=ax_a)
ax2 = [[]] * n
for k in range(n):
    ax2[k] = plt.Subplot(fig, grid2[k, 0])
    fig.add_subplot(ax2[k])
    ax2[k].plot(t_a, pulses[0][:, k], 'k')
    sns.despine(ax=ax2[k], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    ax2[k].set_yticks([])
    ax2[k].set_xticks([])
    ax2[k].set_xlim(0, x_lim)

t_p = np.linspace(0, len(pulses[1][:, n-1])/fs, len(pulses[1][:, n-1])) * 1000
ax2[n-1].set_xlabel('Active')
grid3 = matplotlib.gridspec.GridSpecFromSubplotSpec(n, 1, subplot_spec=ax_p)
ax3 = [[]] * n
for k in range(n):
    ax3[k] = plt.Subplot(fig, grid3[k, 0])
    fig.add_subplot(ax3[k])
    ax3[k].plot(t_p, pulses[1][:, n-k-1], 'k')
    sns.despine(ax=ax3[k], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    ax3[k].set_yticks([])
    ax3[k].set_xticks([])
    ax3[k].set_xlim(0, x_lim)

sns.despine(ax=ax2[n-1], top=True, right=True, left=True, bottom=False, offset=5, trim=False)
sns.despine(ax=ax3[n-1], top=True, right=True, left=True, bottom=False, offset=5, trim=False)
ax2[n-1].set_xlabel('Time [ms]')
ax3[n-1].set_xlabel('Time [ms]')
ax2[n-1].set_xticks([0, 0.1, x_lim])
ax3[n-1].set_xticks([0, 0.1, x_lim])

# MATRIX PLOTS
# Plot on axes
XX_AP, YY_AP = np.meshgrid(np.linspace(0, n_passive, n_passive+1), np.linspace(0, n_active, n_active+1))
XX_AA, YY_AA = np.meshgrid(np.linspace(0, n_active, n_active+1), np.linspace(0, n_active, n_active+1))
XX_PP, YY_PP = np.meshgrid(np.linspace(0, n_passive, n_passive+1), np.linspace(0, n_passive, n_passive+1))

rasterize_it = True

ax[1].pcolormesh(XX_AA, YY_AA, matrix_data[0], cmap=cm_selena, rasterized=rasterize_it, vmin=0, vmax=1)
ax[1].set_xlabel('Active')
ax[1].set_ylabel('Active')

ax[2].pcolormesh(XX_PP, YY_PP, matrix_data[1], cmap=cm_selena, rasterized=rasterize_it, vmin=0, vmax=1)
ax[2].set_xlabel('Passive')
ax[2].set_ylabel('Passive')

ax[3].pcolormesh(XX_AP, YY_AP, matrix_data[2], cmap=cm_selena, rasterized=rasterize_it, vmin=0, vmax=1)
ax[3].set_xlabel('Passive')
ax[3].set_ylabel('Active')

sns.despine(ax=ax[1], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
sns.despine(ax=ax[2], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
sns.despine(ax=ax[3], top=True, right=True, left=True, bottom=True, offset=None, trim=False)


ax[1].set_xticks(np.arange(1.5, len(XX_AA)+0.5, 4))
ax[1].set_xticklabels([2, 6, 10, 14, 18, 22, 26, 30])
ax[1].set_yticks(np.arange(1.5, len(YY_AA)+0.5, 4))
ax[1].set_yticklabels([2, 6, 10, 14, 18, 22, 26, 30])

ax[2].set_xticks(np.arange(1.5, len(XX_PP)+0.5, 4))
ax[2].set_xticklabels([2, 6, 10, 14, 18, 22, 26, 30])
ax[2].set_yticks(np.arange(1.5, len(YY_PP)+0.5, 4))
ax[2].set_yticklabels([2, 6, 10, 14, 18, 22, 26, 30])

ax[3].set_xticks(np.arange(1.5, len(XX_AP)+0.5, 4))
ax[3].set_xticklabels([2, 6, 10, 14, 18, 22, 26])
ax[3].set_yticks(np.arange(1.5, len(YY_AP)+0.5, 4))
ax[3].set_yticklabels([2, 6, 10, 14, 18, 22, 26, 30])

# Spectrogram
freq = f/1000
t_freq = t * 1000
ax[0].pcolormesh(t_freq, freq, db, cmap=cm_selena, vmin=-100, vmax=0, shading='gouraud', rasterized=rasterize_it)
ax[0].set_ylim(0, 150)
ax[0].set_yticks([0, 50, 100, 150])
ax[0].set_xlim(0, np.round(np.max(t_freq)))
ax[0].set_xlabel('Time [ms]')
ax[0].set_ylabel('Freq. [kHz]')
sns.despine(ax=ax[0], top=True, right=True, left=True, bottom=True, offset=None, trim=False)

# ax[0].axhline(100, color='r')
# ax[0].plot([0, 20], [80, 80], 'r')

# Color Bars
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cm_selena, norm=norm, orientation='vertical')
norm = matplotlib.colors.Normalize(vmin=-100, vmax=0)
cb_spec = matplotlib.colorbar.ColorbarBase(ax_cb_spec, cmap=cm_selena, norm=norm, orientation='vertical')
# ax_cb_spec.set_xticks([0, -50, -100])
cb_spec.set_ticks([0, -50, -100])
cb_spec.set_ticklabels([' 0', '-50', '-100'])
cb_spec.outline.set_visible(False)
cb1.outline.set_visible(False)

# Sub Lables
ax[0].text(-0.3, 1.4, subfig_caps_labels[0], transform=ax[0].transAxes, size=subfig_caps,
              color='black')
ax2[0].text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2[0].transAxes, size=subfig_caps,
              color='black')
ax[1].text(label_x_pos, 1.2, subfig_caps_labels[2], transform=ax[1].transAxes, size=subfig_caps,
              color='black')

# Axis labels
# fig.text(0.05, 0.9, 'Freq. [kHz]', ha='center', fontdict=None, rotation=90)
# fig.text(0.05, 0.7, 'Active', ha='center', fontdict=None, rotation=90)
# fig.text(0.05, 0.4, 'Passive', ha='center', fontdict=None, rotation=90)
# fig.text(0.05, 0.2, 'Passive', ha='center', fontdict=None, rotation=90)
fig.text(0.85, 0.5, 'Cross Correlation Value', ha='center', va='center', fontdict=None, rotation=-90)
fig.text(0.48, 0.82, 'dB', ha='center', va='center', fontdict=None, rotation=-90)
fig.text(0.2, 0.65, 'Active', ha='center', va='center', fontdict=None, rotation=0)
fig.text(0.36, 0.65, 'Passive', ha='center', va='center', fontdict=None, rotation=0)

figname = data_path + species + '/' + animal + '/' + recording_nr + '/call_nr_' + str(call_nr) + '/' + species + '_' + \
          recording_nr + '_' + str(call_nr) + '.pdf'

fig.savefig(figname, dpi=400)
plt.close(fig)
print('Done')
exit()
