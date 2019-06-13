#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:12:03 2019

@author: DFSCHMIDT
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import basic_functions as bf
import scipy
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import indicators_calculation as ic



def scatter_plot(coord_agg, point_size = 0.1, ax=None):
    """
    Plots the cloud of points of the aggregate
    """
    # plot the cloud of points as a figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(coord_agg[:, 0], coord_agg[:, 1],
               coord_agg[:, 2], marker='o', s=point_size)
    #ax.view_init(elev=100., azim=90.)
    return ax


def radii_histogram(coord):
    """
    Plot the histogram of radii from the ellpsoid test image
    Needs its coordinates (3D array)
    Return the radii histogram plot
    """
    # compute histogram of radii
    radii = np.sqrt(np.einsum('ai,ai->a', coord, coord))
    hist, bins = np.histogram(radii, bins=100)
    bins = (bins[1:] + bins[:-1])/2.

    # plot the histogram graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('radii')
    ax.set_ylabel('# of points')
    ax.plot(bins, hist)

    return


def bbox_plot(coord_agg, bbox, npoints_bbox=20, point_size = 0.00001, ax=None):
    """
    Plots the cloud of points of the aggregate and its bounding box
    Needs the coordinates of the aggregate (3D array),
    the optimized rotation angles between x and y and
    the number of points in each edge of the bounding box (npoints_bbox)
    """
    # plot the cloud of points as a figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.scatter(coord_agg[:, 0], coord_agg[:, 1],
               coord_agg[:, 2], marker='o', s=point_size)

    # draw bounding box
    ROT = bf.rotation(bbox['angles'])
    coord_rot = np.dot(coord_agg, ROT)

    min_x = min(coord_rot[:, 0])
    max_x = max(coord_rot[:, 0])
    min_y = min(coord_rot[:, 1])
    max_y = max(coord_rot[:, 1])
    min_z = min(coord_rot[:, 2])
    max_z = max(coord_rot[:, 2])

    x = np.ndarray.tolist(np.linspace(min_x, max_x, num=npoints_bbox))
    y = np.ndarray.tolist(np.linspace(min_y, max_y, num=npoints_bbox))
    z = np.ndarray.tolist(np.linspace(min_z, max_z, num=npoints_bbox))

    MIN_X = np.ndarray.tolist(np.full((npoints_bbox), min_x))
    MAX_X = np.ndarray.tolist(np.full((npoints_bbox), max_x))
    MIN_Y = np.ndarray.tolist(np.full((npoints_bbox), min_y))
    MAX_Y = np.ndarray.tolist(np.full((npoints_bbox), max_y))
    MIN_Z = np.ndarray.tolist(np.full((npoints_bbox), min_z))
    MAX_Z = np.ndarray.tolist(np.full((npoints_bbox), max_z))

    X = x + x + x + x + MIN_X + MAX_X + MIN_X + \
        MAX_X + MIN_X + MAX_X + MIN_X + MAX_X
    Y = MIN_Y + MAX_Y + MIN_Y + MAX_Y + y + \
        y + y + y + MIN_Y + MIN_Y + MAX_Y + MAX_Y
    Z = MIN_Z + MIN_Z + MAX_Z + MAX_Z + MIN_Z + \
        MIN_Z + MAX_Z + MAX_Z + z + z + z + z

    plot_coord = np.dot(
        np.c_[X, Y, Z],
        bf.rotation(-bbox['angles'], 'yx')
    )
    ax.scatter(plot_coord[:, 0], plot_coord[:, 1],
               plot_coord[:, 2], marker='o', s=1, color='r')
    # ax.view_init(elev=100., azim=45)

    return


def plot_ellipsoid(ellipsoid, ax=None):
    "simply draw an ellipsoid"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    rx, ry, rz = ellipsoid['a'], ellipsoid['b'], ellipsoid['c']
    # Radii corresponding to the coefficients:

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4,
                      color='r', linewidths=.6)
    return

def fit_ellipsoid_plot(coord_agg, ellipsoid, point_size=0.1):
    """
    Plots the cloud of points of the aggregate and its bounded or
    bounding ellipsoid
    Needs the coordinates of the aggregate (3D array),
    the half-axes of the fitting ellipsoid and
    the number of points composing the ellipsoid (npoints_ellipsoid)
    """
    # plot the cloud of points as a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.scatter(coord_agg[:, 0], coord_agg[:, 1],
               coord_agg[:, 2], marker='o', s=point_size)

    # draw ellipsoid
    plot_ellipsoid(ellipsoid, ax=ax)
    #ax.view_init(elev=-1, azim=90)
    return ax

def map_plot(zi, scale_maxvalue=0.01):
    """
    Plot color map 
    Neeeds X, Y, Z as 1D arrays
    The scale_maxvalue allows to set the extremum of the z axis
    """
    ax= plt.figure()
    # I control the range of my colorbar by removing data 
    # outside of my range of interest
    zmin = -scale_maxvalue
    zmax = scale_maxvalue
    # Only in case of contourf:
    #zi[(zi<zmin) | (zi>zmax)] = None

    # Create the contour plot (seismic, rainbow)
    ax = plt.imshow(zi[1:-1,1:-1], vmin = zmin, vmax = zmax, cmap=plt.cm.seismic, origin= {'lower'})
    plt.colorbar()  
    plt.show()
    return ax

def roughness_map_plot(distance, scale_maxvalue=0.004, sigma=5):
    """
    Plot the roughness map for an aggregate
    Neeeds the roughness distance composed at least by:
        -Angles theta is on the x axis
        -Angles phi is on the y axis
        -Roughness distances is on the z axis (colormap)
    The scale_maxvalue allows to set the extremum of the z axis
    """
    axe1 = plt.figure()
    axe2 = plt.figure()
    axe3 = plt.figure()
    
    # create x-y points to be used in heatmap
    xi = np.linspace(distance[:, 0].min(), distance[:, 0].max(), 2000)
    yi = np.linspace(distance[:, 1].min(), distance[:, 1].max(), 1000)
    
    # Z is a matrix of x-y values
    zi = griddata((distance[:, 0], distance[:, 1]), distance[:, 2], (xi[None,:], yi[:,None]), method='cubic')
    zi_gaussian = roughness_gaussian_filter(zi, sigma)['zi_gaussian'] 
    gaussian_filtered_roughness = roughness_gaussian_filter(zi, sigma)['gaussian_filtered_roughness']

    axe1 = map_plot(zi, scale_maxvalue)    
    axe2 =  map_plot(zi_gaussian, scale_maxvalue)  
    axe3 =  map_plot(gaussian_filtered_roughness, scale_maxvalue/50.)

    return {'original_img': axe1, 
            'gaussian_img': axe2,
            'gaussian_filtered_roughness_img': axe3,
            'gaussian_filtered_roughness': gaussian_filtered_roughness
            }

def roughness_gaussian_filter(zi, sigma=5):
    zi_gaussian = scipy.ndimage.gaussian_filter(zi, sigma)
    gaussian_filtered_roughness = zi-zi_gaussian
    return {'zi_gaussian': zi_gaussian,
            'gaussian_filtered_roughness': gaussian_filtered_roughness
            }

def roughness_distance_histogram(distance, bins=100):
    """
    Plot the histogram of roughness distance from the aggregate
    Needs the roughness distance
    Return the roughness distance histogram plot
    """
    # compute histogram of radii
    hist, bins = np.histogram(distance, bins)
    bins = (bins[1:] + bins[:-1])/2.

    # plot the histogram graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('distance')
    ax.set_ylabel('# of points')
    ax.plot(bins, hist)
    return
