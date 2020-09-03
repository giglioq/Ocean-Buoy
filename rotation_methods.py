# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:17:43 2020

@author: starlord
"""
import pandas as pd
import numpy as np


def rotation_matrix_from_vectors(vec1, vec2):
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
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
        a = [df.iloc[i]['Ax'],df.iloc[i]['Ay'],df.iloc[i]['Az']]
        b = [df.iloc[i]['Bx'],df.iloc[i]['By'],df.iloc[i]['Bz']]
        # calculate a rotation matrix a onto b
        R = rotation_matrix_from_vectors(a,b)
        # apply the rotation to the acceleration vector
        rotated_a = R.dot(a)
        # fill out the new data frame
        rotated_df.iloc[i]['Ax'] = rotated_a[0]
        rotated_df.iloc[i]['Ay'] = rotated_a[1]
        rotated_df.iloc[i]['Az'] = rotated_a[2]
    return rotated_df
    
    
    
def cartesian_to_spherical(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan(y/x)
    theta = np.arctan(np.sqrt(x**2+y**2)/z)
    return r, phi, theta



    
