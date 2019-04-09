# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:34:31 2019

@author: DFSCHMIDT
"""
#!/usr/bin/env python3

import numpy as np
from numpy.random import random_sample as rand # random_sample returns floats in the interval ]0.0;1.0]


def create_ellipsoid(npoints, a, b, c):
    """
    Create ellipsoid with a, b and c as half-axes and npoints the number of points (has to be an integer)
    return an array with shape (npoints, 3) 
    """
    points = np.zeros((npoints, 3)) # create an array of zeros from size npoints x 3
    theta = rand(npoints)*2.*np.pi # create a vector of npoints angles theta ]0;2pi]
    phi = rand(npoints)*np.pi  # create a vector of npoints angles phi ]0;pi]
    points[:, 0] = a*np.cos(theta)*np.sin(phi) # x
    points[:, 1] = b*np.sin(theta)*np.sin(phi) # y
    points[:, 2] = c*np.cos(phi) # z 
    return points


def add_noise(points, amplitude): 
    """
    Add noise to a 3D array with a given amplitude
    """
    npoints = points.shape[0] # collects the number of points from the points array in create_sphere
    dX = (rand((npoints, 3)) - 1.)*amplitude # create an array of npoints x 3 with floats btw ]-1;0] multiplied by an amplitude
    points += dX # add noise to points


def compute_center(points): 
    """
    Compute center of a cloud of points (3D array expected)
    """
    center = np.average(points, axis=0)
    return center


def rotation(theta, phi, order):
    """"
    Compute rotation array with theta angle rotating along the x axis and phi rotating along the y axis 
    Order is 'xy' = rotation along x first and then along y, or 'yx' = rotation along y first and then along x
    """
    Rx = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    Ry = [[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0], [np.sin(phi), 0, np.cos(phi)]]
    if order == 'xy':
        M = np.dot(Rx, Ry)
    elif order == 'yx':
        M = np.dot(Ry, Rx)
    return M