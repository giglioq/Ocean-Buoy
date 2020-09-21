# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:23:43 2020

@author: starlord
"""

import serial
import time
import pandas as pd


t0 = time.time()
# initialize a list with header labels
lst = [b't,Ax,Ay,Az,Wx,Wy,Wz,Bx,By,Bz,phi,theta,psi,Altitude\n']

# create the serial port object
port = serial.Serial('COM11', 115200, timeout=0.5)

i = 0

while (i < 1024):
    port.write(b's')  # handshake with Arduino
    if (port.inWaiting()):  # if the arduino replies
        value = port.readline()  # read the reply

        # t = time.localtime()#get the currernt time (uses my laptop which is currently in Central Time)
        # current_time = time.strftime("%H:%M:%S", t)#format
        # lst.append(current_time.encode())#append at start of new line (used for index)
        lst.append(bytes(str((time.time() - t0) * 1000), 'utf-8'))
        lst.append(b',')  # csv
        # append the serial print from the arduino, this ends with a println
        lst.append(value)

        print(i)
        time.sleep(0.01)
        i += 1  # used for development so i dont have infnt data

port.close()


with open('data/sampleDataEuler.csv', 'wb') as f:  # obviously cut and paste from the internet
    for line in lst:
        f.write(line)  # write each line into the csv, it dectects , and \n


# create a dataframe using the file we just made
df = pd.read_csv('data/sampleDataEuler.csv',
                 index_col=0, error_bad_lines=False)

# take a peek into the dataframe
print(df.head())


# plot against time index col
# df.plot()

# or we can look at an individual column like this
# df['h'].plot()


# the best option by far
# df.plot(subplots=True,figsize=(11,17),layout=(3,4))
