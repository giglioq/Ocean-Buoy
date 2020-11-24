
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.integrate import cumtrapz
import numbers


class Quaternion:

    def __init__(self, w, x=None, y=None, z=None):
        q = []
        if isinstance(w, Quaternion):
            q = w.q
        elif isinstance(w, np.ndarray):
            if len(w) == 4:
                q = w
            elif len(w) == 3:
                q = np.append(0, w)

        elif x is not None and y is not None and z is not None:
            q = [w, x, y, z]

        elif isinstance(w,list):
            q=w

        self.q=q
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.q)
        self.q = self.q/norm
        self.w = self.q[0]
        self.x = self.q[1]
        self.y = self.q[2]
        self.z = self.q[3]

    def to_array(self):
        return self.q

    def conjugate(self):
        return Quaternion(self.q * np.array([1.0, -1.0, -1.0, -1.0]))

    def product(self, p):
        return Quaternion(np.array([
            self.w * p.w - self.x * p.x - self.y * p.y - self.z * p.z,
            self.w * p.x + self.x * p.w + self.y * p.z - self.z * p.y,
            self.w * p.y - self.x * p.z + self.y * p.w + self.z * p.x,
            self.w * p.z + self.x * p.y - self.y * p.x + self.z * p.w,

        ]))

    def __str__(self):
        return "({:-.4f} {:+.4f}i {:+.4f}j {:+.4f}k)".format(self.w, self.x, self.y, self.z)

    def __add__(self, p):
        return Quaternion(self.to_array() + p.to_array())

    def __sub__(self, p):
        return Quaternion(self.to_array() - p.to_array())

    def __mul__(self, p):
        if isinstance(p, Quaternion):
            return self.product(p)
        elif isinstance(p, numbers.Number):
            q = self.q * p
            return Quaternion(q)

    def __rmul__(self, p):
        if isinstance(p, Quaternion):
            return self.product(p)
        elif isinstance(p, numbers.Number):
            q = self.q * p
            return Quaternion(q)

    def to_angles(self) -> np.ndarray:
        """
        Return corresponding Euler angles of quaternion.
        Given a unit quaternions :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        its corresponding Euler angles [WikiConversions]_ are:
        .. math::
            \\begin{bmatrix}
            \\phi \\\\ \\theta \\\\ \\psi
            \\end{bmatrix} =
            \\begin{bmatrix}
            \\mathrm{atan2}\\big(2(q_wq_x + q_yq_z), 1-2(q_x^2+q_y^2)\\big) \\\\
            \\arcsin\\big(2(q_wq_y - q_zq_x)\\big) \\\\
            \\mathrm{atan2}\\big(2(q_wq_z + q_xq_y), 1-2(q_y^2+q_z^2)\\big)
            \\end{bmatrix}
        Returns
        -------
        angles : numpy.ndarray
            Euler angles of quaternion.
        """
        phi = np.arctan2(2.0 * (self.w * self.x + self.y * self.z),
                         1.0 - 2.0 * (self.x**2 + self.y**2))
        theta = np.arcsin(2.0 * (self.w * self.y - self.z * self.x))
        psi = np.arctan2(2.0 * (self.w * self.z + self.x * self.y),
                         1.0 - 2.0 * (self.y**2 + self.z**2))
        return np.array([phi, theta, psi])

    def as_rotation_matrix(self):
        R = np.array([
            [self.w**2 + self.x**2 - self.y**2 - self.z**2, 2 * (self.x * self.y - self.w * self.z),
             2 * (self.w * self.y + self.x * self.z)],
            [2 * (self.x * self.y + self.w * self.z),self.w**2 - self.x**2
             + self.y**2 - self.z**2, 2 * (self.y * self.z - self.w * self.x)],
            [2 * (self.x * self.z - self.w * self.y), 2 *
             (self.w * self.x + self.y * self.z), self.w**2 - self.x**2
             - self.y**2 + self.z**2 ]
        ])
        return R

    def rotate_vector(self, v):
        #V = [0,v[0],v[1],v[2]]
       # V = Quaternion(V)
        R = self.as_rotation_matrix()
        return R @ v



def madgwickUpdate(q, a, g, m, dt=0.01, gain=0.41):
    if g is None or not np.linalg.norm(g) > 0:
        return q

    qEst = 0.5 * (q * Quaternion(g)).to_array()

    if np.linalg.norm(a) == 0:
        return q

    a_norm = np.linalg.norm(a)
    a = a / a_norm
    if m is None or not np.linalg.norm(m) > 0:
        return q

    h = q * (Quaternion(m) * q.conjugate())
    bx = np.linalg.norm([h.x, h.y])
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
    q = q + Quaternion(qEst * dt)
    return q


class Madgwick:

    def __init__(self, acc=None, gyr=None, mag=None, **kwargs):
        self.acc = acc
        self.gyr = gyr
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.dt = kwargs.get('dt', 1.0 / self.frequency)
        self.q0 = kwargs.get('q0', Quaternion(np.array([1, 0, 0, 0])))
        if self.acc is not None and self.gyr is not None:
            self.Q, self.earthAcc = self.compute()

    def compute(self):
        N = len(self.acc)
        Q = []
        Q.append(self.q0)
        for i in range(1, N):
            Q.append(madgwickUpdate(
                Q[i - 1], self.acc[i], self.gyr[i], self.mag[i], dt=self.dt))

        earthAcc = np.zeros((N, 3))

        for i in range(1, N):
            earthAcc[i] = Q[i].conjugate().rotate_vector(self.acc[i])
        print('computed earth frame acceleration vector')
        return Q, earthAcc


def get_position(acc, dt):
    vx = cumtrapz(acc[:, 0], dx=dt)
    vy = cumtrapz(acc[:, 1], dx=dt)
    vz = cumtrapz(acc[:, 2], dx=dt)
    fig, ax = plt.subplots()
    ax.plot(vx, label='vx')
    ax.plot(vy, label='vy')
    ax.plot(vz, label='vz')
    ax.legend()
    x = cumtrapz(vx, dx=dt)
    y = cumtrapz(vy, dx=dt)
    z = cumtrapz(vz, dx=dt)
    print('got position')
    return pd.DataFrame([x, y, z]).T


def co_spectrum(A1, A2):
    return A1.real * A2.real + A1.imag * A2.imag


def quad_spectrum(A1, A2):
    return A1.real * A2.imag - A1.imag * A2.real


def directional_distribution(a1, a2, b1, b2, theta):
    return ((1 / np.pi) * (0.5 + a1 * np.cos(theta) +
                           b1 * np.sin(theta) +
                           a2 * np.cos(2 * theta) +
                           b2 * np.sin(2 * theta)))


def frequency_analysis(position, dt=0.01):
    position = position.to_numpy()
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    A_xf = fftpack.fft(x)
    A_yf = fftpack.fft(y)
    A_zf = fftpack.fft(z)
    print('fast fourier transformed')
    A = [A_xf, A_yf, A_zf]

    x_freq = fftpack.fftfreq(len(x), d=dt)  # FFT sample frequency points.
    y_freq = fftpack.fftfreq(len(y), d=dt)
    z_freq = fftpack.fftfreq(len(z), d=dt)
    #print('x freq len: '+ str(len(x_freq)))
    # print((x_freq==z_freq).all())
    C = pd.DataFrame()
    Q = pd.DataFrame()
    coords = ['x', 'y', 'z']

    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            C[coords[i] + coords[j]] = co_spectrum(A[i], A[j])
    # print(C.info())
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            Q[coords[i] + coords[j]] = quad_spectrum(A[i], A[j])

    # These are not used for waves. These describe eddies
    Q['xy'] = 0.0
    Q['yx'] = 0.0

    print(Q.info())

    a1 = Q.xz / (np.sqrt((C.zz + C.yy) * C.zz))
    #print("a1: \n"+str(a1))
    # possibly not negative
    b1 = -Q.yz / ((np.sqrt((C.zz + C.yy) * C.zz)))
    #print("b1: \n"+str(b1))
    a2 = (C.xx - C.yy) / (C.xx + C.yy)
    #print("a2: \n"+str(a2))
    b2 = - 2 * C.xy / (C.xx + C.yy)
    #print("b2: \n"+str(b2))

    theta = np.linspace(0, 2 * np.pi, num=36)

    d = np.zeros([len(theta), len(x_freq)])

    for i, row in enumerate(theta):
        for j, col in enumerate(x_freq):
            d[i, j] = directional_distribution(a1[j], a2[j], b1[j], b2[j], row)
    D = pd.DataFrame(d, index=theta)
    D.index.name = r'$\theta$ in radians'
    return D, pd.DataFrame(A).T, C, Q
