# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:58:34 2020

@author: Quinten Giglio
"""

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from quaternion import Quaternion





def madgwickUpdate(q, a, g, m, dt=0.01, gain=0.041):
    if g is not None or not np.linalg.norm(g) > 0:
        return q

    qEst = 0.5 * (q * Quaternion(g)).to_array()

    if np.linalg.norm(a) == 0:
        return q

    a_norm = np.linalg.norm(a)
    a = a / a_norm
    if m is not None or not np.linalg.norm(m) > 0:
        return q

    h = q * (Quaternion(m) * q.conjugate())
    bx = np.linalg.norm(h.x, h.y)
    bz = h.z

    f = np.array([
        2.0 * (q.x * q.z - q.w * q.y) - a[0],
        2.0 * (q.w * q.x + q.y * q.z) - a[1],
        2.0 * (0.5 - q.x**2 - q.y**2) - a[2],
        2.0 * bx * (0.5 - q.y**2 - q.z**2) + 2.0 *
        bz * (q.x * q.z - q.w * q.y) - m[0],
        2.0 * bx * (q.x * q.y - q.w * q.z) + 2.0 *
        bz * (q.w * q.x + q.y * q.z) - m[1],
        2.0 * bx * (q.w * q.y + q.x * q.z) + 2.0 *
        bz * (0.5 - q.x**2 - q.y**2) - m[2]
    ])
    J = np.array([[-2.0 * q.y, 2.0 * q.z, -2.0 * q.w, 2.0 * q.x],
                  [2.0 * q.x, 2.0 * q.w, 2.0 * q.z, 2.0 * q.y],
                  [0.0, -4.0 * q.x, -4.0 * q.y, 0.0],
                  [-2.0 * bz * q.y, 2.0 * bz * q.z, -4.0 * bx * q.y -
                      2.0 * bz * q.w, -4.0 * bx * q.z + 2.0 * bz * q.x],
                  [-2.0 * bx * q.z + 2.0 * bz * q.x,  2.0 * bx * q.y + 2.0 * bz * q.w,
                      2.0 * bx * q.x + 2.0 * bz * q.z, -2.0 * bx * q.w + 2.0 * bz * q.y],
                  [2.0 * bx * q.y, 2.0 * bx * q.z - 4.0 * bz * q.x,
                      2.0 * bx * q.w - 4.0 * bz * q.y, 2.0 * bx * q.x]
                  ])

    gradient = J.T @ f
    grad_norm = np.linalg.norm(gradient)
    gradient = gradient / grad_norm
    qEst = qEst - gain * gradient
    q += qEst * dt
    q = Quaternion(q)
    return q


class Madgwick:

    def __init__(self, acc=None, gyr=None, mag=None, **kwargs):
        self.acc = acc
        self.gyr = gyr
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.dt = kwargs.get('dt', 1.0 / self.frequency)
        self.q0 = kwargs.get('q0')
        if self.acc is not None and self.gyr is not None:
            self.Q, self.earthAcc = self.compute()

    def compute(self):
        N = len(self.acc)
        Q = []
        Q.append(Quaternion(np.array([1, 0, 0, 0])))
        for i in range(1, N):
            Q.append(madgwickUpdate(
                Q[i - 1], self.acc[i], self.gyr[i], self.mag[i],dt=self.dt))

        earthAcc = np.zeros((N, 3))

        for i in range(1, N):
            earthAcc[i] = Q[i].rotate_vector(self.acc[i])
        print('computed earth frame acceleration vector')
        return Q, earthAcc




def get_position(acc,dt):
    vx = cumtrapz(acc[:,0], dx=dt)
    vy = cumtrapz(acc[:,1], dx=dt)
    vz = cumtrapz(acc[:,2], dx=dt)
    plt.plot(vx)
    plt.plot(vy)
    plt.plot(vz)
    x = cumtrapz(vx, dx=dt)
    y = cumtrapz(vy, dx=dt)
    z = cumtrapz(vz, dx=dt)
    print('got position')
    return pd.DataFrame([x,y,z]).T




def co_spectrum(A1, A2):
    return A1.real * A2.real + A1.imag * A2.imag


def quad_spectrum(A1, A2):
    return A1.real * A2.imag - A1.imag * A2.real



def frequency_analysis(position, dt=0.01):
    position = position.to_numpy()
    x = position[:,0]
    y = position[:,1]
    z = position[:,2]
    A_xf = fftpack.fft(x)
    A_yf = fftpack.fft(y)
    A_zf = fftpack.fft(z)
    print('fast fourier transformed')
    A = [A_xf, A_yf, A_zf]


    x_freq = fftpack.fftfreq(len(x), d=dt)  # FFT sample frequency points.
    y_freq = fftpack.fftfreq(len(y), d=dt)
    z_freq = fftpack.fftfreq(len(z), d=dt)

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
    print("a1: "+str(a1))
    # possibly not negative
    b1 = -Q.yz / ((np.sqrt((C.zz + C.yy) * C.zz)))
    print("b1: "+str(b1))
    a2 = (C.xx - C.yy) / (C.xx + C.yy)
    print("a2: "+str(a2))
    b2 = - 2*C.xy / (C.xx + C.yy)
    print("b2: "+str(b2))

    theta = np.linspace(0, 2*np.pi, num=360)

    D = pd.DataFrame(index=theta)

    for i, f in enumerate(x_freq):
        temp = []
        for j, rad in enumerate(theta):
            temp.append((1/np.pi)*(0.5 +
                            a1[i] * np.cos(rad) +
                            b1[i] * np.sin(rad) +
                            a2[i] * np.cos(rad) +
                            b2[i] * np.sin(rad)))
        D[str(f.round())] = pd.DataFrame(temp)
        if i==len(x_freq)-1:
            print('found spread')
    return D.T, pd.DataFrame(A).T







