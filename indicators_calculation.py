# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:31:24 2019

@author: DFSCHMIDT
"""

import numpy as np
import basic_functions as bf
import test_image_ellipsoid as tie
import bounding_box as bbox
import plot
import included_ellipsoid as ie
import bounding_ellipsoid as be
import scipy as scipy
from scipy import optimize
from scipy.optimize import minimize


def std_sphericity(tomo_surface, tomo_volume):
    return np.power((np.pi), 1./3.)*np.power((6.*tomo_volume), 2./3.)/tomo_surface

def ab_ratio(a, b):
    return a/b

def ac_ratio(a, c):
    return a/c

def bc_ratio(b, c):
    return b/c

def abc_ratio(a, b, c):
    return a**2/(b*c)

def be_ie_surface_ratio(be_ellipsoid, ie_ellipsoid):
    return  bf.ellipsoid_area(be_ellipsoid)/bf.ellipsoid_area(ie_ellipsoid)

def tomo_ellipsoid_surface_ratio(tomo_surface, ellipsoid):
    """
    Compute the ratio between the tomograph surface and the ellipsoid surface
    Generally, the middle ellipsoid surface is taken 
    (middle ellipsoid = mean ellipsoid between bounding and included ellipsoids)
    """
    return tomo_surface/bf.ellipsoid_area(ellipsoid)

def roughness_distance(aggregate, ellipsoid):
    """
    Compute the distance between an ellipsoid (generally, the middle one) and
    the edges of the aggregate.    
    """
    center = bf.compute_center(aggregate)
    a = ellipsoid['a']
    b = ellipsoid['b']
    c = ellipsoid['c']
    ellipsoid_point = np.zeros((1, 3))
    distance = np.zeros((len(aggregate), 3))
    #aggregate_distance = np.array
    #ellipsoid_distance = np.array
    for i in range(len(aggregate)):
        # Angle calculations
        theta = bf.angle_between([1., 0., 0.], 
                              [aggregate[i, 0] - center[0], 
                               aggregate[i, 1] - center[1], 0.])
        phi = bf.angle_between([0., 0., 1.], 
                            [0., aggregate[i, 1] - center[1], 
                             aggregate[i, 2] - center[2]])
    
        # Eauivalent ellipsoid point calculation
        ellipsoid_point[0, 0] = a*np.cos(theta)*np.sin(phi)
        ellipsoid_point[0, 1] = b*np.sin(theta)*np.sin(phi)
        ellipsoid_point[0, 2] = c*np.cos(phi)
        
        # Distance calculation
        aggregate_distance = np.sqrt((aggregate[i, 0]-center[0])**2 
                                      + (aggregate[i, 1]-center[1])**2
                                      + (aggregate[i, 2]-center[2])**2)
        ellipsoid_distance = np.sqrt((ellipsoid_point[0, 0]-center[0])**2 
                                      + (ellipsoid_point[0, 1]-center[1])**2
                                      + (ellipsoid_point[0, 2]-center[2])**2)
        distance[i, 0] = aggregate_distance - ellipsoid_distance
        distance[i, 1] = theta
        distance[i, 2] = phi
        
    return distance








