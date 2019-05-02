#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:18:09 2019

@author: DFSCHMIDT
"""

import numpy as np
import basic_functions as bf
import bounding_box as bbox
import plot


def included_ellipsoid_optim(coord, tol=1e-3, quiet=True):
    """
    Compute the largest included ellidpsoid of a cloud of points
    Needs the required precision (tol)
    and the coordinates (3D array) of the cloud of points
    """
    # finding optimal bounding box
    bbox_res = bbox.bbox_optim(coord)
    if quiet is False:
        plot.bbox_plot(coord, bbox_res)
        
    # rotation of the cloud of points in the main direction of the bbox
    coord = bf.rotate_aggregate(coord, angles=bbox_res['angles'])
    # plot.bbox_plot(coord, 0., 0., 2)

    # initial a, b, c
    a = (np.sqrt((max(coord[:, 0])-min(coord[:, 0]))**2))/2.
    b = (np.sqrt((max(coord[:, 1])-min(coord[:, 1]))**2))/2.
    c = (np.sqrt((max(coord[:, 2])-min(coord[:, 2]))**2))/2.
    a_before = 0.
    b_before = 0.
    c_before = 0.


    volume_before = 0.
    volume = 4./3.*np.pi*a*b*c
    test = 'false'
    point_inside = 'true'
    
    
    while abs(volume - volume_before) > tol :
        volume_before = volume
        # test if points inside the bounded ellipsoid
        
        for i in range(len(coord)):
            if (coord[i, 0]**2/a**2 + 
                coord[i, 1]**2/b**2 + 
                coord[i, 2]**2/c**2) < 1:
                point_inside = 'true'
                #print('x =', coord[i, 0], 'y =', coord[i, 1], 'z =', coord[i, 2])
                break
            else:
                point_inside = 'false'

        if point_inside == 'true':
            a1 = a_before
            b1 = b_before
            c1 = c_before
            a_before = a
            b_before = b
            c_before = c
            a = a - abs(a1-a)/2.
            b = b - abs(b1-b)/2.
            c = c - abs(c1-c)/2.
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
            print(
                'error, the bbox did not work:'
                ' bbox does not not touch the extrema of the cloud of points')
            break

        volume = 4./3.*np.pi*a*b*c
        """
        print('volume =', volume)
        print('point inside :', point_inside)
        print('New dimensions: a = ', a, 'b = ', b, 'c = ', c)
        print('---')
        """        
        if (quiet is False):
            print('volume =', volume)
            print('a = ', a, 'b = ', b, 'c = ', c)
            plot.fit_ellipsoid_plot(coord, a, b, c, 10000)
               
    return  {'volume': volume,
            'a': a,
            'b': b,
            'c': c,
            'bbox': bbox_res}