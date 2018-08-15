import matplotlib.pyplot as plt
import nixio as nix
import myfunctions
import numpy as np
from IPython import embed

# Data File Name
data_name = '2017-10-26-aa'
data_file = '/home/brehm/data/' + data_name + '/' + data_name + '.nix'

# Open the nix file
f = nix.File.open(data_file, nix.FileMode.ReadOnly)
b = f.blocks[0]

# Get tags
tag = b.tags['MothASongs_0']
mtag = b.multi_tags['MothASongs-moth_song-damped_oscillation*10-7']
# mtag = b.multi_tags['Search-sine_wave-1']

# Get metadata
meta = mtag.metadata.sections[0]
samplingrate = np.round(meta[1])*1000  # in Hz
pulsenumber = len(meta.sections)
MetaData = np.zeros((5, pulsenumber))

# Reconstruct Moth Song Stimulus from Metadata
for i in range(pulsenumber):
    a = meta.sections[i]
    MetaData[0, i] = a["Duration"]
    MetaData[1, i] = a["Tau"]
    MetaData[2, i] = a["Frequency"]
    MetaData[3, i] = a["StartTime"]
    MetaData[4, i] = a["Amplitude"]

stimulus_duration = 0.1
stimulus = np.zeros(samplingrate*stimulus_duration)  # This must also be read from metadata!
data_samplingrate = 100*1000
stimulus_time = np.linspace(0, stimulus_duration, samplingrate*stimulus_duration)
pulse = np.zeros((pulsenumber, samplingrate*MetaData[0, 0]+1))
time = np.linspace(0, MetaData[0, 0], samplingrate*MetaData[0, 0])
for p in range(pulsenumber):
    j = 0
    for t in time:
        pulse[p, j] = np.sin(2*np.pi*MetaData[2, p]*t) * np.exp(-t/MetaData[1, p])
        j += 1
    p0 = int(MetaData[3, p] * samplingrate)
    p1 = int(p0 + (MetaData[0, p] * samplingrate))
    stimulus[p0:p1+1] = pulse[p, :]

# Get Voltage Trace and Spikes corresponding to tag
volt = {}
spikes = {}
volt1 = np.zeros((len(mtag.positions), data_samplingrate*stimulus_duration))
frate = np.zeros((len(mtag.positions), data_samplingrate*stimulus_duration))
for u in range(len(mtag.positions)):
    dummy = myfunctions.get_voltage(mtag,1,u)[:]
    volt1[u, 0:len(dummy)] = dummy
    volt.update({u: myfunctions.get_voltage(mtag,1,u)[:]})  # tagname, tag or multitag?, datasetnumber

    dummy2 = myfunctions.get_spiketimes(mtag, 1, u)[:]
    spikes.update({u: myfunctions.get_spiketimes(mtag, 1, u)[:]})
    _, frate[u] = myfunctions.inst_firing_rate(dummy2-mtag.positions[u], stimulus_duration, 1/data_samplingrate)

MeanVolt = np.mean(volt1, 0)
frate_mean = np.mean(frate, 0)
data_time = np.linspace(0, stimulus_duration, data_samplingrate*stimulus_duration)

# Plotting
plt.subplot(3, 1, 1)
# plt.plot(data_time, volt1[0,:])
for i in range(len(mtag.positions)):
    plt.plot(spikes[i]-mtag.positions[i], np.zeros(len(spikes[i]))+2*(i+1), 'ko')
plt.xlim(0, stimulus_duration)
plt.subplot(3, 1, 2)
plt.plot(data_time*1000, frate_mean, 'k')
plt.xlim(0, stimulus_duration*1000)
plt.subplot(3, 1, 3)
plt.plot(stimulus_time*1000, stimulus, 'k')
plt.xlim(0, stimulus_duration*1000)
plt.show()
