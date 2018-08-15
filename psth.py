import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

tmax = 20
dt = 0.01
spikes = np.array([0.8, 1.3, 6, 6.2, 7, 10, 11.2, 15.7, 16])
time = np.arange(0,tmax,dt)
rate = np.zeros((time.shape[0]))

isi = np.diff(np.insert(spikes,0,0))
inst_rate = 1/isi
spikes_id = np.round(spikes/dt)
spikes_id = np.insert(spikes_id, 0, 0)
spikes_id = spikes_id.astype(int)

for i in range(spikes_id.shape[0]-1):
    rate[spikes_id[i]:spikes_id[i+1]] = inst_rate[i]

print(rate)
plt.plot(time,rate)
plt.show()
