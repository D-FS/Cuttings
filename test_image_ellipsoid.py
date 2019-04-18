#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:46:35 2019

@author: DFSCHMIDT
"""
import numpy as np
import basic_functions as bf


def ellipsoid_test_image(ellipsoid, npoints=1000,
                         noise_amplitude=10., angles=(0., 0.)):
    """
    Create an ellipsoidal test image with npoints and a, b, c as half-axes
    A noise is created with an amplitude = noise_amplitude
    The ellipsoid is rotated along x and along y by angles
    rotation_angle_x and rotation_angle_y
    Return a 3D array with the coordinates of the rotated
    ellispoid with noise
    """

    points = bf.create_ellipsoid(ellipsoid, npoints=npoints)
    bf.add_noise(points, noise_amplitude)
    center = bf.compute_center(points)

    # recenter points = put center to 0
    points = points - center

    # aggregate coordinates xyz after rotation
    test_image_coord = np.dot(points, bf.rotation(angles, order='yx'))

    # recenter test_image_coord = put center to 0
    center = bf.compute_center(test_image_coord)
    test_image_coord = test_image_coord - center

    return test_image_coord
