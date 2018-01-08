import matplotlib.pyplot as plt
import nixio as nix
import numpy as np
from IPython import embed

# Save Variable as numpy array
data = np.random.rand(20, 20)

c = np.loadtxt('testfile.npy')  # This will restore the array from the saved file


# Save Variable as a text file
a = [1,2,3,4,5]
np.savetxt('file_numpy.txt', data, fmt='%.2f')
with open('file_numpy.txt') as file:
    d = file.read()  # This will restore the data as a string variable