# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:58:34 2020

@author: Quinten Giglio
"""

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy.integrate import cumtrapz

from quaternion import Quaternion


pi = np.pi

df = pd.read_csv('data/sampleDataEuler.csv')
df = df.iloc[1:]  # Trim first row
df.dropna(inplace=True)


acc = df.values[:, 1:4]
gyr = df.values[:, 4:7]
mag = df.values[:, 7:10]
samplePeriod = 1 / 256

a = np.array([1, 2, 3, 4])
b = np.array([15, 6, 7, 8])
i = np.array([1, 0, 0, 0])
gain = 1.0
dt = 0.01
quat = Quaternion(i)


def madgwickUpdate(q, a, g, m, st=0.01, gain=0.041):
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
        self.dt = kwargs.get('dt', 1.0/self.frequency)
        self.q0 = kwargs.get('q0')
        if self.acc is not None and self.gyr is not None:
            self.Q , self.earthAcc = self.compute()


    def compute(self):
        N = len(self.acc)
        Q = []
        Q.append(Quaternion(np.array([1, 0, 0, 0])))
        for i in range(1,N):
            Q.append(madgwickUpdate(Q[i-1],self.acc[i],self.gyr[i],self.mag[i]))

        earthAcc = np.zeros((N,3))

        for i in range(1,N):
            earthAcc[i] = Q[i].rotate_vector(acc[i])

        return Q, earthAcc



M = Madgwick(acc=acc, gyr=gyr, mag = mag,q0=quat,dt=dt)



