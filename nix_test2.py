import nixio as nix
import matplotlib.pyplot as plt
import numpy as np

nix_file = nix.File.open('exp4.h5', nix.FileMode.Overwrite)
block = nix_file.create_block("TestBlock", "nix.session")

observations = [0, 0, 5, 20, 45, 40, 28, 12, 2, 0, 1, 0]
categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug','Sep','Oct','Nov', 'Dec']
data = block.create_data_array("observations", "nix.histogram", data=observations)
dim = data.append_set_dimension()
dim.labels = categories

nix_file.close()

file = nix.File.open('exp3.h5', nix.FileMode.Overwrite)




file.close()