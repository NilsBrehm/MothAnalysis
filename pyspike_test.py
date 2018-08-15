import matplotlib.pyplot as plt
import pyspike as spk
from IPython import embed
import numpy as np

# f = '/media/brehm/Data/PySpike/PySpike/test/PySpike_testdata.txt'
# spike_trains = spk.load_spike_trains_from_txt(f, edges=(0, 4000))

spikes = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Calls_spikes.npy').item()
tag_list = np.load('/media/brehm/Data/MasterMoth/figs/2018-02-20-aa/DataFiles/Calls_tag_list.npy')
edges = [0, 2]
d = np.zeros(10)
call_nr = len(tag_list)
k = 1
call_nr = len(tag_list)
sp_between = [[]] * call_nr
sp_within = [[]] * 20

for i in range(call_nr):
    sp_between[i] = spk.SpikeTrain(spikes[tag_list[i]][0], edges)

for i in range(20):
    sp_within[i] = spk.SpikeTrain(spikes[tag_list[50]][i], edges)

m_within = spk.isi_distance_matrix(sp_within)
m_between = spk.isi_distance_matrix(sp_between)

plt.subplot(1,2,1)
plt.imshow(m_within, vmin=0, vmax=1)
plt.colorbar()
plt.title('Within')

plt.subplot(1,2,2)
plt.imshow(m_between, vmin=0, vmax=1)
plt.colorbar()
plt.title('Between')
plt.show()


exit()

d_mean = np.mean(d)
print('Within ' + str(d_mean))
d = np.zeros(10)
for i in range(10):
    sp1 = spk.SpikeTrain(spikes[tag_list[k]][0], edges)
    sp2 = spk.SpikeTrain(spikes[tag_list[i]][0], edges)
    isi_profile = spk.isi_profile(sp1, sp2)
    d[i] = isi_profile.avrg()
    print(tag_list[0] + ' vs ' + tag_list[i])
    print("ISI distance: %.8f" % isi_profile.avrg())
'''
x, y = isi_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile.avrg())
plt.show()
embed()
'''

# sp2 = spk.SpikeTrain(spikes[tag_list[0]][1], edges); isi_profile = spk.isi_profile(sp1, sp2); print("ISI distance: %.8f" % isi_profile.avrg())