from matplotlib import pyplot as plt
import nixio as nix
import numpy as np

freq = np.array([10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 8000, 10000, 15000, 20000, 40000])
volt = np.array([22, 90, 160, 252, 332, 340, 312, 268, 220, 103, 74, 40, 24, 8])
gain = volt/13
db = 20 * np.log(gain)



plt.figure()
plt.plot(freq, volt)
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Voltage [mV]')

plt.figure()
plt.plot(freq, db)
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')

plt.figure()
plt.plot(freq, gain)
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [x-times]')
plt.show()