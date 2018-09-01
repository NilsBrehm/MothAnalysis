import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import nixio as nix
import scipy.io.wavfile as wav
import matplotlib
from scipy import signal as sg
import seaborn as sns
from tqdm import tqdm
import scalebars as sb


def plot_settings():
    # Font:
    # matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10

    # Ticks:
    matplotlib.rcParams['xtick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.size'] = 4
    matplotlib.rcParams['xtick.major.size'] = 4

    # Title Size:
    matplotlib.rcParams['axes.titlesize'] = 10

    # Axes Label Size:
    matplotlib.rcParams['axes.labelsize'] = 10

    # Axes Line Width:
    matplotlib.rcParams['axes.linewidth'] = 1

    # Tick Label Size:
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9

    # Line Width:
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['lines.color'] = 'k'

    # Marker Size:
    matplotlib.rcParams['lines.markersize'] = 2

    # Error Bars:
    matplotlib.rcParams['errorbar.capsize'] = 0

    # Legend Font Size:
    matplotlib.rcParams['legend.fontsize'] = 6

    return matplotlib.rcParams


def tagtostimulus(nix_file, pathname):
    tag_list = np.load(pathname + 'Calls_tag_list.npy')
    f = nix.File.open(nix_file, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    # mtags = {}
    connection = {}
    connection2 = {}
    for k in range(len(tag_list)):
        # mtags.update({tag_list[k]: b.multi_tags[tag_list[k]]})
        mtag = b.multi_tags[tag_list[k]]
        sound_name = mtag.metadata.sections[0][2]
        connection.update({sound_name: tag_list[k]})
        connection2.update({tag_list[k]: sound_name})

    f.close()
    return connection, connection2


def plot_spikes(rec, name, stim_type, volt, spikes, cutoff, trial_nr, t_limit, ID):

    if stim_type is 'series':
        call_name = 'callseries/moths/' + name
        audio_path = '/media/nils/Data/Moth/stimuli_plotting/callseries/moths/'
    if stim_type is 'single':
        call_name = 'naturalmothcalls/' + name
        audio_path = '/media/nils/Data/Moth/stimuli_plotting/naturalmothcalls/'
    else:
        print('Wrong stim type')
        return 0

    p_nix = '/media/nils/Data/Moth/mothdata/' + rec + '/' + rec + '.nix'
    p = '/media/nils/Data/Moth/figs/' + rec + '/DataFiles/'

    c1, c2 = tagtostimulus(p_nix, p)

    stim = c1[call_name]
    call_audio = wav.read(audio_path + name)
    call_volt = volt[stim]
    call_spikes = spikes[stim]
    if trial_nr is None:
        trial_nr = np.random.randint(0, len(call_volt))
    t_audio = np.arange(0, len(call_audio[1])/call_audio[0], 1/call_audio[0])
    fs = 100*1000
    t_volt = np.arange(0, len(call_volt[trial_nr])/fs, 1/fs)

    # print('Trial number: ' + str(trial_nr))

    # Filter volt trace
    nyqst = 0.5 * fs
    lowcut = cutoff[0]
    highcut = cutoff[1]
    low = lowcut / nyqst
    high = highcut / nyqst
    ftype = 'band'
    b, a = sg.butter(cutoff[2], [low, high], btype=ftype, analog=False)
    y = sg.filtfilt(b, a, call_volt[trial_nr])

    call_dur = np.round(t_audio[-1] - 0.075, 2)
    # Plot
    plot_settings()
    if t_limit[1] <= 0:
        t_limit[1] = call_dur

    # Create Grid
    grid = matplotlib.gridspec.GridSpec(nrows=3, ncols=1)
    fig = plt.figure(figsize=(2.9, 1.4))
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])

    ax1.plot(t_audio*1000, call_audio[1], 'k', linewidth=0.5)
    # ax2.plot(t_volt*1000, call_volt[trial_nr], 'b')
    ax2.plot(t_volt*1000, y, 'k', linewidth=0.5)
    for k in range(len(call_spikes)):
        ax3.plot(call_spikes[k]*1000, np.ones(len(call_spikes[k])) + k, 'ks', markersize=0.2, linewidth=0.5)

    # ax1.set_xticks(np.arange(t_limit[0] * 1000, t_limit[1] * 1000, 20))
    # ax2.set_xticks(np.arange(t_limit[0] * 1000, t_limit[1] * 1000, 20))
    # ax3.set_xticks(np.arange(t_limit[0] * 1000, t_limit[1] * 1000, 20))
    # ax3.set_xticks(np.arange(t_limit[0]*1000, t_limit[1]*1000+100, 10), minor=True)

    ax1.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax2.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax3.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    ax1.set_yticks([])
    ax1.set_xticks([])
    ax3.set_xticks([])
    ax2.set_xticks([])
    ax1.set_yticklabels([])
    ax2.set_yticks([])
    # ax3.set_yticks(np.arange(0, len(call_volt)+1, 10))
    ax3.set_yticks([])

    # ax3.set_xlabel('Time [ms]')

    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax3, top=True, right=True, left=True, bottom=True, offset=5, trim=False)

    ob = sb.AnchoredHScaleBar(size=10, label='', loc=1, frameon=False, pad=0.2, sep=4, color="k")
    ax1.add_artist(ob)
    fig.text(0.84, 0.90, '10 ms', ha='center', fontdict=None, size=6)

    # Subplot caps
    subfig_caps = 6
    label_x_pos = 0
    label_y_pos = 0.99
    subfig_caps_labels = ['S', 'V', 'RP', 'd', 'e', 'f', 'g', 'h', 'i']
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.05, right=0.9, wspace=0.2, hspace=0.2)
    # figname = '/media/nils/Data/Moth/figs/' + rec + '/responses' + '/' + name[0:-4] + '_' + \
    #           stim_type + '_trial' + str(trial_nr) + '.pdf'
    # figname = '/media/nils/Data/Moth/figs/' + rec + '/responses/' + stim_type + str(ID) + '.pdf'
    figname = '/media/nils/Data/Moth/Thesis/nilsbrehm/figs/responses/' + stim_type + str(ID) + '.pdf'

    fig.savefig(figname)
    plt.close(fig)
    return 0


# SCRIPT STARTS HERE ===================================================================================================
stims = ['BCI1062_07x07.wav',
         'aclytia_gynamorpha_24x24.wav',
         'agaraea_semivitrea_07x07.wav',
         'carales_12x12_01.wav',
         'chrostosoma_thoracicum_05x05.wav',
         'creatonotos_01x01.wav',
         'elysius_conspersus_11x11.wav',
         'epidesma_oceola_06x06.wav',
         'eucereon_appunctata_13x13.wav',
         'eucereon_hampsoni_11x11.wav',
         'eucereon_obscurum_14x14.wav',
         'gl005_11x11.wav',
         'gl116_05x05.wav',
         'hypocladia_militaris_09x09.wav',
         'idalu_fasciipuncta_05x05.wav',
         'idalus_daga_18x18.wav',
         'melese_12x12_01_PK1297.wav',
         'neritos_cotes_10x10.wav',
         'ormetica_contraria_peruviana_09x09.wav',
         'syntrichura_12x12.wav']

trials = [17, 5, 3, 19, 6, 2, 12, 3, 7, 2, 8, 2, 0, 4, 16, 8, 17, 10, 10, 18]
rec = '2018-02-16-aa'
stim_types = 'single'
c_name = 'melese_11x11_PK1299' + '.wav'
cutoff = [100, 2000, 2]
p = '/media/nils/Data/Moth/figs/' + rec + '/DataFiles/'
volt = np.load(p + 'Calls_voltage.npy').item()
spikes = np.load(p + 'Calls_spikes.npy').item()

for k in tqdm(range(len(stims)), desc='Calls'):
    plot_spikes(rec, stims[k], stim_types, volt, spikes, cutoff, trials[k], [0, 0], ID=k)

print('All plots saved')
