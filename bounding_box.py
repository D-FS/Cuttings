# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:49:37 2019

@author: DFSCHMIDT
"""

#!/usr/bin/env python3

import numpy as np
import scipy as scipy
import basic_functions as bf

def bbox_volume(angles, coord):
    """
    Compute the volume of bouding box of a cloud of points
    for a given rotation of the cloud of points
    Needs the rotation angles (rotation angle along x, rotation angle along y) 
    and the coordinates (3D array) of the cloud of points
    """
    theta, phi = angles
        
    # cloud of points rotation    
    M = bf.rotation(theta, phi, 'xy')
    coord_rot = np.dot(coord, M)
    
    # calculation of the side
    min_x = min(coord_rot[:, 0])
    max_x = max(coord_rot[:, 0])
    min_y = min(coord_rot[:, 1])
    max_y = max(coord_rot[:, 1])
    min_z = min(coord_rot[:, 2])
    max_z = max(coord_rot[:, 2])  
      
    volume = np.sqrt((max_x-min_x)**2)*np.sqrt((max_y-min_y)**2)*np.sqrt((max_z-min_z)**2)
    #volume = (max_x-min_x)*(max_y-min_y)*(max_z-min_z)
    
    return volume


def bbox_optim (coord, initial_guess):
    """
    Minimize the volume of the bounding box of a cloud of points using the L-BFGS-B method
    with respect to the rotation angles of the cloud of points
    Needs the cloud of points coordinates (3D array) and an initial guess of the rotation angle (rotation angle along x, rotation angle along y)
    """
    res = scipy.optimize.minimize(bbox_volume, initial_guess, args=(coord), method='L-BFGS-B')
    print('bouding box optimiztation: \n', res)
    return res


