# -*- coding: utf-8 -*-
import numpy as np
import numbers


class Quaternion:

    def __init__(self, w, x=None, y=None, z=None):
        q = []
        if isinstance(w, Quaternion):
            self.q = w.q
            q = w.q

        elif isinstance(w, np.ndarray):
            if len(w) == 4:
                q = w
            elif len(w) == 3:
                q = np.append(0, w)
            norm = np.linalg.norm(q)
            q = q / norm
            self.q = q
        elif x is not None and y is not None and z is not None:
            self.q = [w, x, y, z]
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

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
            [1 - 2 * (q.y**2 + q.z**2), 2 * (q.x * q.y - q.w * q.z),
             2 * (q.w * q.y + q.x * q.z)],
            [2 * (q.x * q.y + q.w * q.z), 1 - 2 *
             (q.x**2 + q.z**2), 2 * (q.y * q.z - q.w * q.x)],
            [2 * (q.x * q.z - q.w * q.y), 2 *
             (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2)]
        ])
        return R

    def rotate_vector(self, v):
        #V = [0,v[0],v[1],v[2]]
       # V = Quaternion(V)
        R = self.as_rotation_matrix()
        return R @ v


a = np.array([-0.00085769, -0.0404217, 0.29184193, -0.47288709])
b = np.array([1, 0, 0, 0])
q = Quaternion(a)
q2 = Quaternion(b)


v = np.array([0.25557699, 0.74814091, 0.71491841])
