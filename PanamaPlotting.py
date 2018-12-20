import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import scipy.io.wavfile as wav
import scipy as scipy
import myfunctions as mf
import matplotlib
import scipy.io.wavfile as wav

file_name = '/media/brehm/Data/Panama/Recordings/Carales_astur/PK1285/Pk12850017/call_nr_2/matrix_analysis.mat'
audio_name = '/media/brehm/Data/Panama/Recordings/Carales_astur/PK1285/Pk12850017/call_nr_2.wav'
mat_file = scipy.io.loadmat(file_name)
fs, audio_call = wav.read(audio_name)

AvsA = mat_file['MaxCorr_AA']
PvsP = mat_file['MaxCorr_PP']
AvsP = mat_file['MaxCorr_AP']

matrix_data = {0: mat_file['MaxCorr_AA'], 1: mat_file['MaxCorr_PP'], 2: mat_file['MaxCorr_AP']}

# Pulses: 0=Active, 1=Passive
pulses = {0: mat_file['pulses'][0][0][0], 1: mat_file['pulses'][0][0][1]}

# Spectrogram
Nx = 512
FFT = 512
nover = Nx-5
w = scipy.signal.get_window('hann', Nx, fftbins=True)
f, t, Sxx = scipy.signal.spectrogram(audio_call, fs, window=w, nperseg=None, noverlap=nover, nfft=FFT,
                                     detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
db = 20*np.log10(Sxx/np.max(Sxx))


plt.figure(figsize=(10, 2.5))
plt.pcolormesh(t, f, db, cmap='nipy_spectral', vmin=-40, vmax=0, shading='gouraud')
plt.colorbar()
plt.ylim(0, 80000)
plt.show()

mf.plot_settings()
ax = [[]] * 4
grid = matplotlib.gridspec.GridSpec(nrows=47, ncols=11)
fig = plt.figure(figsize=(5.9, 2.4))
ax[0] = plt.subplot(grid[0:10, 0:10])
ax[1] = plt.subplot(grid[12:22, 0:10])
ax[2] = plt.subplot(grid[24:34, 0:10])
ax[3] = plt.subplot(grid[36:46, 0:10])

for k in range(len(ax)-1):
    ax[k].imshow(matrix_data[k])

ax[3].pcolormesh(t, f, Sxx, cmap='jet', vmin=0, vmax=1, shading='gouraud')

embed()
exit()
