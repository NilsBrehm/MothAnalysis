#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Copyright © 2014 German Neuroinformatics Node (G-Node)
 All rights reserved.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted under the terms of the BSD License. See
 LICENSE file in the root of the Project.
 Author: Jan Grewe <jan.grewe@g-node.org>
 This tutorial shows how regulary sampled data is stored in nix-files.
 See https://github.com/G-node/nix/wiki for more information.
"""

import nixio as nix
import numpy as np
import matplotlib.pylab as plt


def plot_data(data_array):
    x_axis = data_array.dimensions[0]
    x = x_axis.axis(data_array.data.shape[0])
    y = data_array.data
    plt.plot(x, y)
    plt.xlabel(x_axis.label + " [" + x_axis.unit + "]")
    plt.ylabel(data_array.label + " [" + data_array.unit + "]")
    plt.title(data_array.name)
    plt.xlim(0, np.max(x))
    plt.ylim((1.1 * np.min(y), 1.1 * np.max(y)))
    plt.show()


if __name__ == "__main__":
    # open a file
    file_name = '/home/brehm/data/2017-06-21-ab/2017-06-21-ab.h5'
    file = nix.File.open(file_name)
    block = file.blocks[0]
    voltage = block.data_arrays['V-1']
    # let's plot the data from the stored information
    plot_data(voltage)
    file.close()
