import scipy.io
from IPython import embed
import myfunctions as mf
import numpy as np
import matplotlib.pyplot as plt

file_name01 = '/media/brehm/Data/Panama/DataForPaper/Castur/PK1285/sorted/17/call_nr_5/call_nr_5_samples.mat'
#file_name02 = '/media/brehm/Data/Panama/DataForPaper/Castur/PK1285/sorted/16/call_nr_5//call_nr_5_samples.mat'
file_name02 = '/media/brehm/Data/Panama/DataForPaper/Castur/PK1285/sorted/17/call_nr_6/call_nr_6_samples.mat'
#file_name03 = '/media/brehm/Data/Panama/DataForPaper/Castur/PK1285/sorted/17/call_nr_8/call_nr_8_samples.mat'
file_name03 = '/media/brehm/Data/Panama/DataForPaper/Melese_incertus/MittenDrin/call_nr_1/call_nr_1_samples.mat'
mat01 = scipy.io.loadmat(file_name01)
mat02 = scipy.io.loadmat(file_name02)
mat03 = scipy.io.loadmat(file_name03)

active01 = mat01['samples'][0][0][0]
passive01 = mat01['samples'][0][0][1]
active02 = mat02['samples'][0][0][0]
passive02 = mat02['samples'][0][0][1]
active03 = mat03['samples'][0][0][0]
passive03 = mat03['samples'][0][0][1]

dt = 480 * 1000

call01 = np.append(active01, passive01) / dt
call02 = np.append(active02, passive02) / dt
call03 = np.append(active03, passive03) / dt

#call01 = np.append(active01, []) / dt
#call02 = np.append(active02, []) / dt
#call03 = np.append(active03, []) / dt

call01 = call01 - call01[0]
call02 = call02 - call02[0]
call03 = call03 - call03[0]

dur01 = np.max(call01)
dur02 = np.max(call02)
dur03 = np.max(call03)
dur = np.min([dur01, dur02, dur03])
# tau = 10  # in ms
# tau = float(input('tau in ms: '))
taus = np.arange(0.5, 20, 0.2)
d = np.zeros(len(taus))
d2 = np.zeros(len(taus))


for i in range(len(taus)):
    d[i] = mf.spike_train_distance(call01, call02, dt, dur, taus[i]/1000, False)
    # print('tau: %s , d = %s' % (str(tau), str(d)))

for i in range(len(taus)):
    d2[i] = mf.spike_train_distance(call01, call03, dt, dur, taus[i]/1000, False)

difference = np.abs(d - d2)
max_distance = np.max(difference)
max_distance_tau = taus[max_distance == difference]

plt.figure()
plt.plot(taus, d, 'k')
plt.plot(taus, d2, 'b')
plt.plot(taus, difference, 'r')
plt.plot(max_distance_tau, max_distance, 'or')
plt.xlabel('tau [ms]')
plt.ylabel('Pulse Train Distance')

plt.figure()
plt.plot(call01, np.zeros(len(call01)), 'ok')
plt.plot(call02, np.zeros(len(call02))+1, 'ok')
plt.plot(call03, np.zeros(len(call03))+2, 'ok')
plt.show()
