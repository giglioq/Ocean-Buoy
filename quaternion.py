# -*- coding: utf-8 -*-
import numpy as np
import numbers


class Quaternion:

    def __init__(self, w, x=None, y=None, z=None):
        q = []
        if isinstance(w, Quaternion):
            q = w.q
        elif isinstance(w, np.ndarray):
            print("test array")
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


q1 = Quaternion(0.55747131, 0.12956903, 0.5736954 , 0.58592763)
q2 = Quaternion(0.49753507, 0.50806522, 0.52711628, 0.4652709)

print(q1*q2)
# expect '(-0.3635 +0.3896i +0.3419j +0.7740k)'


q3 = q1*q2


q = Quaternion(1, 0, 1, 0)
r = [1, 0, 0]
f = [2 ,3, 4]

l = q.conjugate().rotate_vector(r)
