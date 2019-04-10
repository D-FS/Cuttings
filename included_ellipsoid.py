# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:18:09 2019

@author: DFSCHMIDT
"""

#!/usr/bin/env python3

import numpy as np
import basic_functions as bf
import bounding_box as bbox
import plot


def included_ellipsoid_optim(coord, tol):
    """
    Compute the largest included ellidpsoid of a cloud of points
    Needs the required precision (tol) 
    and the coordinates (3D array) of the cloud of points
    """
    # finding optimal bounding box
    bbox_initial_guess =  [0., 0.]
    bbox_res = bbox.bbox_optim(coord, bbox_initial_guess)
    plot.bbox_plot(coord, bbox_res.x[0], bbox_res.x[1], 100)
        
    # rotation of the cloud of points in the main direction of the bbox    
    M = bf.rotation(bbox_res.x[0], bbox_res.x[1], 'xy')
    coord = np.dot(coord, M)
    #plot.bbox_plot(coord, 0., 0., 2)
    
    # initial a, b, c
    a = (np.sqrt((max(coord[:, 0])-min(coord[:, 0]))**2))/2.
    b = (np.sqrt((max(coord[:, 1])-min(coord[:, 1]))**2))/2.
    c = (np.sqrt((max(coord[:, 2])-min(coord[:, 2]))**2))/2.    

    volume_before = 0.
    volume = 4./3.*np.pi*a*b*c
    test = 'false'
     
    while abs(volume - volume_before) > tol:
        volume_before = volume
        # test if points inside the bounded ellipsoid 
        for i in range(len(coord)):
            if (coord[i,0]**2/a**2+coord[i,1]**2/b**2+coord[i,2]**2/c**2)<1:        
                point_inside = 'true'
                break
            else:
                point_inside = 'false'
                      
        if point_inside == 'true':
            a_before = a
            b_before = b
            c_before = c
            a = a*0.95
            b = b*0.95
            c = c*0.95
            test = 'true'
        elif point_inside == 'false' and test == 'true':
            a1 = a_before
            b1 = b_before
            c1 = c_before
            a_before = a
            b_before = b
            c_before = c
            a = a + abs(a1-a)/2.
            b = b + abs(b1-b)/2.
            c = c + abs(c1-c)/2. 
        else:
            print('error, the bbox did not work: bbox does not not touch the extrema of the cloud of points')
            break
        
        volume = 4./3.*np.pi*a*b*c

    plot.fit_ellipsoid_plot(coord, a, b, c, 10000)
    print('a = ', a,'b = ', b,'c = ', c)
    print('volume =', volume)
    return volume, a, b, c




