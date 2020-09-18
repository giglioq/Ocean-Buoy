# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:58:34 2020

@author: Quinten Giglio
"""

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy.integrate import cumtrapz


pi = np.pi

df = pd.read_csv('data/sampleData.csv', index_col=0)
df = df.iloc[1:]  # Trim first row
df.dropna(inplace=True)


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3)
    which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def align_motion_with_mag_field(df):
    """
    # Takes in a pandas data frame
    # Must have columns 'Ax' 'Ay' 'Az' 'Bx' 'By' 'Bz'
    # Rotates the acceleration so that it alings with the magnetic field
    # Returns the corrected acceleration data in a pandas data frame
    """
    rotated_df = pd.DataFrame().reindex_like(df)
    for i in range(len(df)):
        # create matrices of each vector
        a = [df.iloc[i]['Ax'], df.iloc[i]['Ay'], df.iloc[i]['Az']]
        b = [df.iloc[i]['Bx'], df.iloc[i]['By'], df.iloc[i]['Bz']]
        # calculate a rotation matrix a onto b
        R = rotation_matrix_from_vectors(a, b)
        # apply the rotation to the acceleration vector
        rotated_a = R.dot(a)
        # fill out the new data frame
        rotated_df.iloc[i]['Ax'] = rotated_a[0]
        rotated_df.iloc[i]['Ay'] = rotated_a[1]
        rotated_df.iloc[i]['Az'] = rotated_a[2]
    return rotated_df




def smooth(x, window_len=50, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise Exception("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise Exception("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise Exception("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]



df['Bx'] = df['Bx'].rolling(100).mean()
df['By'] = df['By'].rolling(100).mean()
df['Bz'] = df['Bz'].rolling(100).mean()

df['Bx'] = smooth(df['Bx'].values, window_len=700)
df['By'] = smooth(df['By'].values, window_len=700)
df['Bz'] = smooth(df['Bz'].values, window_len=700)


df.dropna(inplace=True)

df.plot(subplots=True, figsize=(15, 15), layout=(3, 5), sharey=False)



rotated_df = align_motion_with_mag_field(df)

dt = rotated_df.index[1]-rotated_df.index[0]


rotated_df['Ax'] = smooth(rotated_df['Ax'].values)
rotated_df['Ay'] = smooth(rotated_df['Ay'].values)
rotated_df['Az'] = smooth(rotated_df['Az'].values)




Vx = cumtrapz(rotated_df['Ax'], dx=dt)
Vy = cumtrapz(rotated_df['Ay'], dx=dt)
Vz = cumtrapz(rotated_df['Az'], dx=dt)

x = cumtrapz(Vx, dx=dt)
y = cumtrapz(Vy, dx=dt)
z = cumtrapz(Vz, dx=dt)

integrals = {
    't': df.index,
    'x': x,
    'y': y,
    'z': z,
    'Vx': Vx,
    'Vy': Vy,
    'Vz': Vz,
    'Ax': rotated_df['Ax'],
    'Ay': rotated_df['Ay'],
    'Az': rotated_df['Az'],
}

motion_df = pd.DataFrame.from_dict(integrals, orient='index')

motion_df = motion_df.transpose()

motion_df.plot(x='t', subplots=True, figsize=(15, 15), layout=(3, 3), sharey=False)


A_xf = fftpack.fft(x)
A_yf = fftpack.fft(y)
A_zf = fftpack.fft(z)
A = [A_xf, A_yf, A_zf]


x_freq = fftpack.fftfreq(len(x), d=dt)  # FFT sample frequency points.
y_freq = fftpack.fftfreq(len(y), d=dt)
z_freq = fftpack.fftfreq(len(z), d=dt)


def co_spectrum(A1, A2):
    return A1.real * A2.real + A1.imag * A2.imag


def quad_spectrum(A1, A2):
    return A1.real * A2.imag - A1.imag * A2.real


C = pd.DataFrame()
Q = pd.DataFrame()
coords = ['x', 'y', 'z']

for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        C[coords[i] + coords[j]] = co_spectrum(A[i], A[j])


for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        Q[coords[i] + coords[j]] = quad_spectrum(A[i], A[j])


# These are not used for waves. These describe eddies
Q['xy'] = 0.0
Q['yx'] = 0.0


a1 = Q.xz / (np.sqrt((C.zz + C.yy) * C.zz))

# possibly not negative
b1 = -Q.yz / ((np.sqrt((C.zz + C.yy) * C.zz)))

a2 = (C.xx - C.yy) / (C.xx + C.yy)

b2 = - 2*C.xy / (C.xx + C.yy)


theta = np.linspace(0, 2*pi, num=6)

D = pd.DataFrame(index=theta)


for i, f in enumerate(x_freq):
    temp = []
    for j, rad in enumerate(theta):
        temp.append((1/pi)*(0.5 +
                            a1[i] * np.cos(rad) +
                            b1[i] * np.sin(rad) +
                            a2[i] * np.cos(rad) +
                            b2[i] * np.sin(rad)))
    D[f] = temp
