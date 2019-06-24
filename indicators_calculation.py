# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:31:24 2019

@author: DFSCHMIDT
"""

import numpy as np
import basic_functions as bf
import plot
import math

def indicators(aggregate, bounding_ellipsoid, middle_ellipsoid, included_ellipsoid, 
               tomo_surface, tomo_volume, scale_maxvalue=0.004, sigma=5):
    """
    Compute all indicators
    """
    
    a = bounding_ellipsoid['a']
    b = bounding_ellipsoid['b']
    c = bounding_ellipsoid['c']
    
    print('Aggregate standard sphericity = ', std_sphericity(tomo_surface, tomo_volume))
    print('Bounding box and ellispoids ratios :')    
    print('a/b =', ab_ratio(a, b))
    print('a/c =', ac_ratio(a, c))
    print('b/c =', ab_ratio(b, c))
    print('a^2/bc =', abc_ratio(a, b, c))
    print('Bounding ellipsoid surface / Included ellipsoid surface =', 
          be_ie_surface_ratio(bounding_ellipsoid, included_ellipsoid))
    print('Aggregate tomographed surface / Bounding ellipsoid surface =', 
          tomo_ellipsoid_surface_ratio(tomo_surface, bounding_ellipsoid))
    print('Aggregate tomographed surface / Middle ellipsoid surface =', 
          tomo_ellipsoid_surface_ratio(tomo_surface, middle_ellipsoid))
    print('Aggregate tomographed surface / Included ellipsoid surface =', 
          tomo_ellipsoid_surface_ratio(tomo_surface, included_ellipsoid))
    print('Roughness map (middle ellipsoid): ')
    distance = roughness_distance(aggregate, middle_ellipsoid)
    print('Mean absolute roughness distance =', roughness_stats(distance)['roughness_mean'])
    print('Roughness distance min and max', roughness_stats(distance)['distance_min_max'])
    map1 = plot.roughness_map_plot(distance, scale_maxvalue, sigma)
    print('Mean absolute gaussian filtered roughness distance =', 
          gaussian_roughness_stats(map1['gaussian_filtered_roughness'])['gaussian_roughness_mean'])
    print('Gaussian filtered roughness distance min and max', 
          gaussian_roughness_stats(map1['gaussian_filtered_roughness'])['gaussian_distance__min_max'])
    print('Roughness histogram (unfiltered)')
    plot.roughness_distance_histogram(distance[:, 2])
    #print('Roughness histogram (filtered)')
    #plot.roughness_distance_histogram(map1['gaussian_filtered_roughness'], bins=10)
    
    return {'aggregate_standard_sphericity': std_sphericity(tomo_surface, tomo_volume),
            'a/b': ab_ratio(a, b),
            'a/c': ac_ratio(a, c),
            'b/c': ab_ratio(b, c),
            'a^2/bc': abc_ratio(a, b, c),
            'bounding_ellipsoid_ surface/included_ellipsoid_surface': 
                be_ie_surface_ratio(bounding_ellipsoid, included_ellipsoid),
            'aggregate_tomographed_surface/bounding_ellipsoid_surface': 
                tomo_ellipsoid_surface_ratio(tomo_surface, bounding_ellipsoid),
            'aggregate_tomographed_surface/middle_ellipsoid_surface': 
                tomo_ellipsoid_surface_ratio(tomo_surface, middle_ellipsoid),
            'aggregate_tomographed_surface/included_ellipsoid_surface': 
                tomo_ellipsoid_surface_ratio(tomo_surface, included_ellipsoid),
            'roughness_distance': roughness_distance(aggregate, middle_ellipsoid),
            'gaussian_filtered_roughness': map1['gaussian_filtered_roughness']
            }

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

def be_ie_surface_ratio(bounding_ellipsoid, included_ellipsoid):
    return  bf.ellipsoid_area(bounding_ellipsoid)/bf.ellipsoid_area(included_ellipsoid)

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
    the edges of the aggregate    
    """
    center = bf.compute_center(aggregate)
    a = ellipsoid['a']
    b = ellipsoid['b']
    c = ellipsoid['c']
    ellipsoid_point = np.zeros((len(aggregate), 3))
    distance = np.zeros((len(aggregate), 5))

    for i in range(len(aggregate)):
        # Angle calculations
        theta = bf.angle_between_2D([1., 0.], 
                              [aggregate[i, 0], 
                               aggregate[i, 1]])
        phi = bf.angle_between([0., 0., 1.], 
                            [aggregate[i, 0], 
                             aggregate[i, 1], 
                             aggregate[i, 2]])
        
        alpha = math.atan2(a*np.sin(theta), b*np.cos(theta))
        if alpha < 0. :
            alpha += 2.*np.pi 
            
        beta = math.atan2(c*np.sin(phi), 
                          (np.cos(phi)*(np.sqrt((a*np.cos(alpha))**2+(b*np.sin(alpha))**2))))

        # Equivalent ellipsoid point calculation
        ellipsoid_point[i, 0] = a*np.cos(alpha)*np.sin(beta)
        ellipsoid_point[i, 1] = b*np.sin(alpha)*np.sin(beta)
        ellipsoid_point[i, 2] = c*np.cos(beta)

        # Distance calculation
        aggregate_distance = np.sqrt((aggregate[i, 0]-center[0])**2 
                                      + (aggregate[i, 1]-center[1])**2
                                      + (aggregate[i, 2]-center[2])**2)
        ellipsoid_distance = np.sqrt((ellipsoid_point[i, 0]-center[0])**2 
                                      + (ellipsoid_point[i, 1]-center[1])**2
                                      + (ellipsoid_point[i, 2]-center[2])**2)
        distance[i, 0] = theta
        distance[i, 1] = phi
        distance[i, 2] = aggregate_distance - ellipsoid_distance
        distance[i, 3] = alpha
        distance[i, 4] = beta
        
    return distance

def roughness_stats(distance):
    """
    Compute the mean of the absolute roughness distance and the 
    extrema (min and max) of the angles (theta, phi, alpha and beta
    and the distance
    Expect [5, n] array shape
    """ 
    theta_min_max = [np.min(distance[:, 0]), np.max(distance[:, 0])]
    phi_min_max = [np.min(distance[:, 1]), np.max(distance[:, 1])]
    distance_min_max = [np.min(distance[:, 2]), np.max(distance[:, 2])]
    alpha_min_max = [np.min(distance[:, 3]), np.max(distance[:, 3])]
    beta_min_max = [np.min(distance[:, 4]), np.max(distance[:, 4])]
    roughness_mean = np.mean(np.abs(distance[:, 2]))
    return {'roughness_mean': roughness_mean,
            'distance_min_max': distance_min_max,
            'theta_min_max': theta_min_max,
            'phi_min_max': phi_min_max,
            'alpha_min_max': alpha_min_max,
            'beta_min_max': beta_min_max
            }
            

def gaussian_roughness_stats(gaussian_distance):
    """
    Compute the mean and the extrema of the absolute roughness distance 
    for a gaussian filtered image
    Expect 1D array
    """
    gaussian_roughness_mean = np.nanmean(np.abs(gaussian_distance))
    gaussian_distance__min_max = [np.nanmin(gaussian_distance), np.nanmax(gaussian_distance)]
    return {'gaussian_roughness_mean': gaussian_roughness_mean,
            'gaussian_distance__min_max': gaussian_distance__min_max
            }