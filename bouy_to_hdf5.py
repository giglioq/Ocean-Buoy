# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:31:59 2020

@author: starlord
"""

import h5py

import serial
import time
import numpy as np
import pandas as pd


# We should be able to get by if we only record the start
# time and the delta t inbetween each measurement because they happen 
# on regular intervals




# using pandas
#wavedata = pd.HDFStore('wavedata.h5')


# f = h5py.File("name.hdf5", "w") # New file overwriting any existing file
# f = h5py.File("name.hdf5", "r") # Open read-only (must exist)
# f = h5py.File("name.hdf5", "r+") # Open read-write (must exist)
# f = h5py.File("name.hdf5", "a") # Open read-write (create if doesn't exist)
wavedata = h5py.File('wavedata.hdf5','w')

arr = np.ones((10,10))

wavedata['dataset'] = arr


wavedata.close()


#using with handles closing file
with h5py.File('wavedata.hdf5', 'r+') as f:
    dat = f['dataset']
    print(dat[()])