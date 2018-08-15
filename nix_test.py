import nixio as nix
import matplotlib.pyplot as plt
import numpy as np

nix_file = nix.File.open('example.h5', nix.FileMode.Overwrite)
block = nix_file.create_block("Test block", "nix.session")

# create an empty DataArray to store 2x1000 values
some_numpy_array =  np.random.randn(2, 1000)
data = block.create_data_array("my data", "nix.sampled", data=some_numpy_array)


nix_file.close()


file_name = 'exp2.h5'

# create a new file overwriting any existing content
file = nix.File.open(file_name, nix.FileMode.Overwrite)

sample_interval = 0.001 # s
sinewave = np.sin(np.arange(0, 1.0, sample_interval) * 2 * np.pi)
data = block.create_data_array("sinewave","nix.regular_sampled",data=sinewave)
data.label = "voltage"
data.unit = "mV"
# define the time dimension of the data
dim = data.append_sampled_dimension(sample_interval)
dim.label = "time"
dim.unit = "s"

file.close()