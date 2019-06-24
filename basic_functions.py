#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:34:31 2019

@author: DFSCHMIDT
"""

import numpy as np
import math
from scipy.special import ellipkinc, ellipeinc
from numpy.random import random_sample as rand


def create_ellipsoid(ellipsoid, npoints=1000):
    """
    Create ellipsoid with a, b and c as half-axes and
    npoints the number of points (has to be an integer)
    return an array with shape (npoints, 3)
    """
    # create an array of zeros from size npoints x 3
    points = np.zeros((npoints, 3)) 
    # create a vector of npoints angles theta ]0;2pi]
    theta = rand(npoints)*2.*np.pi
    # create a vector of npoints angles phi ]0;pi]
    phi = rand(npoints)*np.pi

    a = ellipsoid['a']
    b = ellipsoid['b']
    c = ellipsoid['c']
    points[:, 0] = a*np.cos(theta)*np.sin(phi)  # x
    points[:, 1] = b*np.sin(theta)*np.sin(phi)  # y
    points[:, 2] = c*np.cos(phi)  # z
    return points

def ellipsoid_area(ellipsoid):
    """
    Compute the surface of a define ellipsoid with its half-axes (a, b, c)
    """
    c, b, a = sorted([ellipsoid['a'], ellipsoid['b'], ellipsoid['c']])
    if a == b == c:
        area = 4*np.pi*a**2
    else:
        phi = np.arccos(c/a)
        m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
        temp = ellipeinc(phi, m)*np.sin(phi)**2 + ellipkinc(phi, m)*np.cos(phi)**2
        area = 2*np.pi*(c**2 + a*b*temp/np.sin(phi))
    return area

def mid_ellipsoid(bounding_ellipsoid, included_ellipsoid):
    """
    Compute the middle ellipsoid
    It is the average ellipsoid between the bounding one and the included one
    """
    a = (bounding_ellipsoid['a'] + included_ellipsoid['a'])/2.
    b = (bounding_ellipsoid['b'] + included_ellipsoid['b'])/2.
    c = (bounding_ellipsoid['c'] + included_ellipsoid['c'])/2.
    volume = 4./3.*np.pi*a*b*c
    return {'volume': volume,
            'a': a,
            'b': b,
            'c': c}

def add_noise(points, amplitude):
    """
    Add noise to a 3D array with a given amplitude
    """
    # collects the number of points from the points array in create_sphere
    npoints = points.shape[0]
    # create an array of npoints x 3
    # with floats btw ]-1;0] multiplied by an amplitude
    dX = (rand((npoints, 3)) - 1.)*amplitude
    points += dX  # add noise to points
    return

def compute_center(points):
    """
    Compute center of a cloud of points (3D array expected)
    """
    center = np.average(points, axis=0)
    return center

def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.  
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_2D(v1, v2):
    """ 
    Returns the angle [0, 2pi[ in radians between 2D vectors 'v1' and 'v2'::

            >>> angle_between((1, 0), (0, 1))
            1.5707963267948966
           
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]      # dot product between [x1, y1] and [x2, y2]
    det = v1[0]*v2[1] - v1[1]*v2[0]      # determinant
    angle = np.arctan2(det, dot)        # atan2(y, x) or atan2(sin, cos)
    if angle < 0. :
        angle += 2.*np.pi 
    return angle

def angle_between_3D_inplane(v1, v2, vn):
    """ 
    Returns the angle in radians between 3D vectors 'v1' and 'v2' 
    in a known plane with normal unit (to be normalized if not!) vector vn::

            >>> angle_between((1, 0), (0, 1))
            1.5707963267948966
           
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]     # dot product 
    det = v1[0]*v2[1]*vn[2] + v2[0]*vn[1]*v1[2] + vn[0]*v1[1]*v2[2] - v1[2]*v2[1]*vn[0] - v2[2]*vn[1]*v1[0] - vn[2]*v1[1]*v2[0]      # determinant
    angle = math.atan2(det, dot)        # atan2(y, x) or atan2(sin, cos)
    return angle

def rotation(angles, order='xy'):
    """"
    Compute rotation array with theta angle rotating along the x axis
    and phi rotating along the y axis
    Order is 'xy' = rotation along x first and then along y,
    or 'yx' = rotation along y first and then along x
    """

    theta, phi = angles
    Rx = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
          [0, np.sin(theta), np.cos(theta)]]
    Ry = [[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0],
          [np.sin(phi), 0, np.cos(phi)]]
    if order == 'xy':
        M = np.dot(Rx, Ry)
    elif order == 'yx':
        M = np.dot(Ry, Rx)
    return M


def rotate_aggregate(coords, mat=None, angles=None, **kwargs):
    """
    Apply a rotation onto an aggregate
    """

    if mat is None:
        if angles is None:
            raise RuntimeError('Need angles or matrix to rotate')
        mat = rotation(angles, **kwargs)
    coords_rot = np.dot(coords, mat)
    return coords_rot
