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
import os
import myfunctions as mf
import scipy.io as sio


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


def get_directories(data_name):
    data_files_path = os.path.join('..', 'figs', data_name, 'DataFiles', '')
    figs_path = os.path.join('..', 'figs', data_name, '')
    nix_path = os.path.join('..', 'mothdata', data_name, data_name)
    thesis_figs = os.path.join('..', 'Thesis/nilsbrehm/figs/responses/')
    audio_path = os.path.join('..', 'stimuli_plotting/')
    path_names = [data_name, data_files_path, figs_path, nix_path, thesis_figs, audio_path]
    return path_names


def remove_large_spikes(x, spike_times, mph_percent, method):
    # Remove large spikes
    fs = 100*1000
    if method == 'max':
        if spike_times.any():
            mask = np.ones(len(spike_times), dtype=bool)
            idx = spike_times * fs
            a = x[idx.astype('int')]
            idx_a = a >= np.max(a) * mph_percent
            mask[idx_a] = False
            marked = spike_times[idx_a]
            spike_times = spike_times[mask]

    if method == 'std':
        if spike_times.any():
            mask = np.ones(len(spike_times), dtype=bool)
            idx = spike_times * fs
            a = x[idx.astype('int')]
            a = a / np.max(a)
            idx_a = a >= np.mean(a) + np.std(a) * mph_percent
            mask[idx_a] = False
            marked = spike_times[idx_a]
            spike_times = spike_times[mask]
        else:
            marked = np.nan

    return spike_times, marked


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


def plot_spikes(rec, name, stim_type, volt, spikes, cutoff, trial_nr, t_limit, ID, path_names):

    if stim_type is 'series':
        call_name = 'callseries/moths/' + name
        audio_path = path_names[5] + 'callseries/moths/'

    if stim_type is 'single':
        call_name = 'naturalmothcalls/' + name
        audio_path = path_names[5] + 'naturalmothcalls/'

    if stim_type is 'bats_single':
        call_name = 'batcalls/' + name
        audio_path = path_names[5] + 'batcalls/'

    if stim_type is 'bats_series':
        call_name = 'callseries/bats/' + name
        audio_path = path_names[5] + 'callseries/bats/'

    # p_nix = '/media/nils/Data/Moth/mothdata/' + rec + '/' + rec + '.nix'
    p_nix = path_names[3] + '.nix'

    # p = '/media/nils/Data/Moth/figs/' + rec + '/DataFiles/'
    p = path_names[1]

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

    # print(str(ID) + ' - ' + name + ' Trial number: ' + str(trial_nr))

    if stim_type is 'series':
        marked = [[]] * len(call_volt)
        # Remove large spikes
        for i in range(len(call_volt)):
            call_spikes[i], marked[i] = remove_large_spikes(call_volt[i], call_spikes[i], mph_percent=10, method='std')

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
    if stim_type is 'bats_single':
        fig = plt.figure(figsize=(2.9, 1.2))
    else:
        fig = plt.figure(figsize=(2.9, 1.4))
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])

    if stim_type is 'bats_single' or stim_type is 'bats_series':
        ax1.plot(t_audio[0:len(call_audio[1])]*1000, call_audio[1], 'k', linewidth=0.5)
    else:
        ax1.plot(t_audio * 1000, call_audio[1], 'k', linewidth=0.5)
    ax2.plot(t_volt*1000, y, 'k', linewidth=0.5)
    fs = 100*1000
    if stim_type is 'series':
        mared_idx = marked[trial_nr] * fs
        ax2.plot(marked[trial_nr]*1000, y[mared_idx.astype('int')], 'rx', linewidth=0.5, markersize=0.25)

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
    # figname = '/media/nils/Data/Moth/Thesis/nilsbrehm/figs/responses/' + stim_type + str(ID) + '.pdf'
    # figname = path_names[4] + 'test/' + stim_type + str(ID) + '.pdf'
    figname = path_names[2] + 'responses/' + stim_type + str(ID) + '.png'
    fig.savefig(figname)
    plt.close(fig)
    return 0


def plot_volt(rec, name, stim_type, volt, spikes, spikes_raw, valleys, spikes_old, cutoff, trial_nr, t_limit, ID, path_names):

    if stim_type is 'series' or stim_type is 'bonus_series':
        pulsetrains = sio.loadmat('/media/nils/Data/Moth/CallStats/CallSeries_Stats/samples.mat')['samples'][0]
        call_name = 'callseries/moths/' + name
        audio_path = path_names[5] + 'callseries/moths/'

    if stim_type is 'single' or stim_type is 'bonus_single':
        pulsetrains = sio.loadmat('/media/nils/Data/Moth/CallStats/samples.mat')['samples'][0]
        call_name = 'naturalmothcalls/' + name
        audio_path = path_names[5] + 'naturalmothcalls/'

    if stim_type is 'bats_single':
        call_name = 'batcalls/' + name
        audio_path = path_names[5] + 'batcalls/'

    if stim_type is 'bats_series':
        call_name = 'callseries/bats/' + name
        audio_path = path_names[5] + 'callseries/bats/'

    # p_nix = '/media/nils/Data/Moth/mothdata/' + rec + '/' + rec + '.nix'
    p_nix = path_names[3] + '.nix'

    # p = '/media/nils/Data/Moth/figs/' + rec + '/DataFiles/'
    p = path_names[1]

    c1, c2 = tagtostimulus(p_nix, p)

    stim = c1[call_name]
    call_audio = wav.read(audio_path + name)
    call_volt = volt[stim]
    call_spikes = spikes[stim]
    call_spikes_raw = spikes_raw[stim]
    call_old = spikes_old[stim]
    call_vals = valleys[stim]
    if trial_nr is None:
        trial_nr = np.random.randint(0, len(call_volt))
    t_audio = np.arange(0, len(call_audio[1])/call_audio[0], 1/call_audio[0])
    fs = 100*1000
    t_volt = np.arange(0, len(call_volt[trial_nr])/fs, 1/fs)

    # print(str(ID) + ' - ' + name + ' Trial number: ' + str(trial_nr))

    # # Pulse trains
    # if stim_type is 'series' or stim_type is 'single':
    #     call_pulsetrain = pulsetrains[ID][0]
    #     for n_trials in range(len(call_spikes)):
    #         pulse_distance = np.zeros(len(call_spikes[n_trials]))
    #         for n_spikes in range(len(call_spikes[n_trials])):
    #             abs_diff = abs(call_pulsetrain - call_spikes[n_trials][n_spikes])
    #             pulse_distance[n_spikes] = np.min(abs_diff)
    #             # idx_pulse = np.where(abs_diff == np.min(abs_diff))
    #         idx_pulse = pulse_distance <= 0.02
    #         call_spikes[n_trials] = call_spikes[n_trials][idx_pulse]

    # Filter volt trace
    nyqst = 0.5 * fs
    lowcut = cutoff[0]
    highcut = cutoff[1]
    low = lowcut / nyqst
    high = highcut / nyqst
    ftype = 'band'
    b, a = sg.butter(cutoff[2], [low, high], btype=ftype, analog=False)

    # Filter
    y = [[]] * len(call_volt)
    for i in range(len(call_volt)):
        y[i] = sg.filtfilt(b, a, call_volt[i])

    call_dur = np.round(t_audio[-1] - 0.075, 2)
    # Plot
    plot_settings()
    if t_limit[1] <= 0:
        t_limit[1] = call_dur

    # Create Grid
    trial_count = len(call_volt)
    grid = matplotlib.gridspec.GridSpec(nrows=trial_count+5, ncols=1)
    fig = plt.figure(figsize=(5.9, 10))
    plt.title(call_name)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    ax4 = plt.subplot(grid[3, 0])
    ax5 = plt.subplot(grid[4, 0])

    if stim_type is 'bats_single' or stim_type is 'bats_series':
        ax1.plot(t_audio[0:len(call_audio[1])] * 1000, call_audio[1], 'k', linewidth=0.5)
    else:
        ax1.plot(t_audio * 1000, call_audio[1], 'k', linewidth=0.5)

    # Raster plot
    for k in range(len(call_spikes)):
        ax2.plot(call_spikes[k]*1000, np.ones(len(call_spikes[k])) + k, 'rs', markersize=0.2, linewidth=0.5)
        ax3.plot(call_spikes_raw[k]*1000, np.ones(len(call_spikes_raw[k])) + k, 'gs', markersize=0.2, linewidth=0.5)
        ax4.plot(call_vals[k]*1000, np.ones(len(call_vals[k])) + k, 'bs', markersize=0.2, linewidth=0.5)
        ax5.plot(call_old[k]*1000, np.ones(len(call_old[k])) + k, 'ms', markersize=0.2, linewidth=0.5)

    for j in range(trial_count):
        ax = plt.subplot(grid[j+5, 0])
        t_volt = np.arange(0, len(call_volt[j]) / fs, 1 / fs)
        ax.plot(t_volt*1000, y[j], 'k', linewidth=0.5)
        # marked_idx = marked[j] * fs
        marked_idx = call_spikes[j] * fs
        vals_idx = call_vals[j] * fs
        old_idx = call_old[j] * fs
        raw_idx = call_spikes_raw[j] * fs

        ax.plot(call_spikes[j]*1000, y[j][marked_idx.astype('int')], 'ro', linewidth=1, markersize=1.5)
        ax.plot(call_spikes_raw[j]*1000, y[j][raw_idx.astype('int')], 'go', linewidth=1, markersize=0.8)
        ax.plot(call_vals[j]*1000, y[j][vals_idx.astype('int')], 'bo', linewidth=1, markersize=0.7)
        ax.plot(call_old[j]*1000, y[j][old_idx.astype('int')], 'mx', linewidth=1, markersize=0.4)

        ax.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True, offset=False, trim=False)

    ax1.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])

    ax2.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticklabels([])

    ax3.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax3.set_xticklabels([])
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticklabels([])

    ax4.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax4.set_xticklabels([])
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticklabels([])

    ax5.set_xlim(t_limit[0] * 1000, t_limit[1] * 1000)
    ax5.set_xticklabels([])
    ax5.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticklabels([])

    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax3, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax4, top=True, right=True, left=True, bottom=True, offset=False, trim=False)
    sns.despine(ax=ax5, top=True, right=True, left=True, bottom=True, offset=False, trim=False)

    ob = sb.AnchoredHScaleBar(size=10, label='', loc=1, frameon=False, pad=0.2, sep=4, color="k")
    ax1.add_artist(ob)
    fig.text(0.84, 0.90, '10 ms', ha='center', fontdict=None, size=6)

    # Subplot caps
    subfig_caps = 4
    label_x_pos = 0
    label_y_pos = 0.99
    subfig_caps_labels = [name, 'New Spike Detection', 'Raw Spike Detection', 'Valleys', 'Old Spike Detection', 'f', 'g', 'h', 'i']
    ax1.text(label_x_pos, label_y_pos, subfig_caps_labels[0], transform=ax1.transAxes, size=subfig_caps,
             color='black')
    ax2.text(label_x_pos, label_y_pos, subfig_caps_labels[1], transform=ax2.transAxes, size=subfig_caps,
             color='black')
    ax3.text(label_x_pos, label_y_pos, subfig_caps_labels[2], transform=ax3.transAxes, size=subfig_caps,
             color='black')
    ax4.text(label_x_pos, label_y_pos, subfig_caps_labels[3], transform=ax4.transAxes, size=subfig_caps,
             color='black')
    ax5.text(label_x_pos, label_y_pos, subfig_caps_labels[4], transform=ax5.transAxes, size=subfig_caps,
             color='black')

    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.05, right=0.9, wspace=0.2, hspace=0.2)
    # figname = path_names[4] + 'test/' + stim_type + str(ID) + '.pdf'
    figname = path_names[2] + 'responses/VoltageTraces/VoltageTraces_' + stim_type + str(ID) + '.pdf'
    fig.savefig(figname)
    plt.close(fig)
    return 0


# SCRIPT STARTS HERE ===================================================================================================
# ss = ['single', 'series', 'bats_single', 'bats_series', 'bonus_series', 'bonus_single']
ss = ['single', 'series']

rec = '2018-02-16-aa'
path_names = get_directories(rec)
cutoff = [100, 2000, 2]

protocol_name = 'Calls'
th_factor = 4
window = None
mph_percent = 0.8
mf.spike_times_calls(path_names, protocol_name, show=False, save_data=True, th_factor=th_factor, filter_on=True,
                     window=window, mph_percent=mph_percent, selection=True)

volt = np.load(path_names[1] + 'Calls_voltage.npy').item()
spikes = np.load(path_names[1] + 'Calls_spikes.npy').item()
spikes_raw = np.load(path_names[1] + 'Calls_spikes_raw.npy').item()
valleys = np.load(path_names[1] + 'Calls_valleys.npy').item()
spikes_old = np.load(path_names[1] + 'backup/Calls_spikes.npy').item()

for q in range(len(ss)):  # Loop through stimulus types
    # stim_types = 'bats_series'
    stim_types = ss[q]
    if stim_types is 'single':
        # rec = '2018-02-16-aa'  # good recs for single calls
        call_limit = 0.19
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

    if stim_types is 'series':
        # rec = '2018-02-16-aa'  # good recs for call series
        call_limit = 0
        stims = ['A7838.wav',
                 'BCI1348.wav',
                 'Chrostosoma_thoracicum.wav',
                 'Creatonotos.wav',
                 'Eucereon_appunctata.wav',
                 'Eucereon_hampsoni.wav',
                 'Eucereon_maia.wav',
                 'GL005.wav',
                 'Hyaleucera_erythrotelus.wav',
                 'Hypocladia_militaris.wav',
                 'PP241.wav',
                 'PP612.wav',
                 'PP643.wav',
                 'Saurita.wav',
                 'Uranophora_leucotelus.wav',
                 'carales_PK1275.wav',
                 'melese_PK1300_01.wav']

    if stim_types is'bats_single':
        call_limit = 0.1
        # rec = '2018-02-20-aa'  # good recs for bats single
        stims = ['Barbastella_barbastellus_1_n.wav',
                 'Eptesicus_nilssonii_1_s.wav',
                 'Myotis_bechsteinii_1_n.wav',
                 'Myotis_brandtii_1_n.wav',
                 'Myotis_nattereri_1_n.wav',
                 'Nyctalus_leisleri_1_n.wav',
                 'Nyctalus_noctula_2_s.wav',
                 'Pipistrellus_pipistrellus_1_n.wav',
                 'Pipistrellus_pygmaeus_2_n.wav',
                 'Rhinolophus_ferrumequinum_1_n.wav',
                 'Vespertilio_murinus_1_s.wav']

    if stim_types is 'bats_series':
        call_limit = 0.3
        # rec = '2018-02-16-aa'  # good recs for bats series
        stims = ['Barbastella_barbastellus_1_n.wav',
                 'Myotis_bechsteinii_1_n.wav',
                 'Myotis_brandtii_1_n.wav',
                 'Myotis_nattereri_1_n.wav',
                 'Nyctalus_leisleri_1_n.wav',
                 'Nyctalus_noctula_2_s.wav',
                 'Pipistrellus_pipistrellus_1_n.wav',
                 'Pipistrellus_pygmaeus_2_n.wav',
                 'Rhinolophus_ferrumequinum_1_n.wav',
                 'Vespertilio_murinus_1_s.wav']

    if stim_types is 'bonus_series':
        call_limit = 4
        stims = ['Chrostosoma_thoracicum_02.wav',
                 'melese_PK1297_01.wav',
                 'melese_PK1298_01.wav',
                 'melese_PK1298_02.wav',
                 'melese_PK1299_01.wav']

    if stim_types is 'bonus_single':
        call_limit = 0.19
        stims = ['aclytia_gynamorpha_24x24_02.wav',
                 'agaraea_semivitrea_06x06.wav',
                 'carales_11x11_01.wav',
                 'carales_11x11_02.wav',
                 'carales_12x12_02.wav',
                 'carales_13x13_01.wav',
                 'carales_13x13_02.wav',
                 'carales_19x19.wav',
                 'chrostosoma_thoracicum_04x04.wav',
                 'chrostosoma_thoracicum_04x04_02.wav',
                 'chrostosoma_thoracicum_05x05_02.wav',
                 'creatonotos_01x01_02.wav',
                 'elysius_conspersus_05x05.wav',
                 'elysius_conspersus_08x08.wav',
                 'epidesma_oceola_05x05.wav',
                 'epidesma_oceola_05x05_02.wav',
                 'eucereon_appunctata_11x11.wav',
                 'eucereon_appunctata_12x12.wav',
                 'eucereon_hampsoni_07x07.wav',
                 'eucereon_hampsoni_08x08.wav',
                 'eucereon_obscurum_10x10.wav',
                 'gl005_04x04.wav',
                 'gl005_05x05.wav',
                 'gl116_04x04.wav',
                 'gl116_04x04_02.wav',
                 'hypocladia_militaris_03x03.wav',
                 'hypocladia_militaris_09x09_02.wav',
                 'idalu_fasciipuncta_05x05_02.wav',
                 'melese_11x11_PK1299.wav',
                 'melese_12x12_PK1299.wav',
                 'melese_13x13_PK1299.wav',
                 'melese_14x14_PK1297.wav',
                 'neritos_cotes_07x07.wav',
                 'neritos_cotes_10x10_02.wav',
                 'ormetica_contraria_peruviana_06x06.wav',
                 'ormetica_contraria_peruviana_08x08.wav',
                 'ormetica_contraria_peruviana_30++.wav',
                 'syntrichura_07x07.wav',
                 'syntrichura_09x09.wav']


    trials = [17, 5, 3, 19, 6, 2, 12, 3, 7, 2, 8, 2, 0, 4, 16, 8, 17, 10, 10, 18]  # good trials for single calls
    # for i in range(len(stims)):
    #     print(str(i) + ': ' + stims[i][0:-4])
    # exit()

    for kk in tqdm(range(len(stims)), desc=stim_types):
        plot_volt(rec, stims[kk], stim_types, volt, spikes, spikes_raw, valleys, spikes_old, cutoff, None, [0, call_limit], ID=kk, path_names=path_names)

    # change = [5, 6, 7, 9]
    # trials = [0, 1, 1, 10]
    # change_stims = np.array(stims)[change]
    # for kk in tqdm(range(len(change_stims)), desc='Calls'):
    #     plot_spikes(rec, change_stims[kk], stim_types, volt, spikes, cutoff, trials[kk], [0, 0], ID=change[kk], path_names=path_names)
    # for kk in tqdm(range(len(stims)), desc='Calls'):
    #     plot_spikes(rec, stims[kk], stim_types, volt, spikes, cutoff, None, [0, call_limit], ID=kk, path_names=path_names)
    #
    # print('All plots saved')
