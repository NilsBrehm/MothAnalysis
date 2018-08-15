import numpy as np
import matplotlib.pyplot as plt
import myfunctions


# t_samples = np.linspace(0.0,1.0,1000)
#
# samples = [square_func(i) for i in t_samples]
#
# print(samples)
# plt.plot(t_samples,samples)
# plt.show()

# t, F = myfunctions.square_wave([0,1],1000,1)
# print(t)
# print(F)
# plt.plot(t,F)
# plt.show()

# -- Square Wave -------------------------------------------------------------------------------------------------------
# Unit: time in msec, frequency in Hz, dutycycle in %, value = 0: disabled
N               = 1000  # sample count
freq            = 0
dutycycle       = 0
period          = 200
pulseduration   = 50

if dutycycle != 0:
    pulseduration = period*(dutycycle/100)
elif freq != 0:
    period = (1 / freq) * 1000
sw = np.arange(N) % period < pulseduration # Returns bool array (True=1, False=0)
plt.plot(sw)
plt.show()