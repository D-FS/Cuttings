#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:49:37 2019

@author: DFSCHMIDT
"""

import numpy as np
from scipy.optimize import minimize
import basic_functions as bf


def bbox_volume(coords, angles=(0., 0.)):
    """
    Compute the volume of bouding box of a cloud of points
    for a given rotation of the cloud of points
    Needs the rotation angles (rotation angle along x, rotation angle along y)
    and the coordinates (3D array) of the cloud of points
    """

    angles = np.array(angles)
    # cloud of points rotation
    coords_rot = bf.rotate_aggregate(coords, angles=angles)

    # calculation of the side
    _min = coords_rot.min(axis=0)
    _max = coords_rot.max(axis=0)
    lengths = np.sqrt((_max-_min)**2)
    volume = lengths[0]*lengths[1]*lengths[2]

    return volume


def angle_optim(coord, initial_guess=(0, 0), method='Nelder-Mead', quiet=True):
    """
    Minimize the volume of the bounding box of a cloud of points
    using the L-BFGS-B method
    with respect to the rotation angles of the cloud of points
    Needs the cloud of points coordinates (3D array) and an initial guess
    of the rotation angle (rotation angle along x, rotation angle along y)
    """
    
    if method == 'Nelder-Mead':
        options = {'maxfev': 1000}
    else:
        options = None
    
    res = minimize(
        lambda angle, coords: bbox_volume(coords, angle),
        initial_guess, args=(coord), method = method, options = options)
    if quiet is False:
        print('bounding box optimization: \n', res)
    if not res.success:
        raise RuntimeError('optimization failed: ' + str(res))
    return res.x, res.fun


def compute_bbox(coords):
    return {'angles': np.array([0., 0.]),
            'volume': bbox_volume(coords)}


def bbox_optim(coords, **kwargs):
    angles, volume = angle_optim(coords, **kwargs)
    volume2 = bbox_volume(coords, angles)
    if volume != volume2:
        raise RuntimeError('internal error')
    return {'angles': angles,
            'volume': volume}
