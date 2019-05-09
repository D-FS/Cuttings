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
from scipy.interpolate import griddata
import indicators_calculation as calc


def scatter_plot(coord_agg, ax=None):
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
               coord_agg[:, 2], marker='o', s=0.000001)
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
    ax.plot(bins, hist)

    return


def bbox_plot(coord_agg, bbox, npoints_bbox=20, ax=None):
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
               coord_agg[:, 2], marker='o', s=0.00001)

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
    "simply drawf an ellipsoid"

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

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def fit_ellipsoid_plot(coord_agg, ellipsoid):
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
               coord_agg[:, 2], marker='o', s=0.01)

    # draw ellipsoid
    plot_ellipsoid(ellipsoid, ax=ax)
    #ax.view_init(elev=-1, azim=90)
    
    return ax

def roughness_map_plot(aggregate, ellipsoid):
    
    distance = calc.roughness_distance(aggregate, ellipsoid)
    
    ax = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    X, Y, Z, = np.array([]), np.array([]), np.array([])
    for i in range(len(distance)):
            X = np.append(X, distance[i, 0])
            Y = np.append(Y, distance[i, 1])
            Z = np.append(Z, distance[i, 2])
    
    # create x-y points to be used in heatmap
    xi = np.linspace(X.min(), X.max(), 1000)
    yi = np.linspace(Y.min(), Y.max(), 1000)
    
    # Z is a matrix of x-y values
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')
    

    # I control the range of my colorbar by removing data 
    # outside of my range of interest
    zmin = -12
    zmax = 12
    zi[(zi<zmin) | (zi>zmax)] = None
    
    # Create the contour plot
    ax = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.seismic,
                      vmax=zmax, vmin=zmin)
        
    #ax = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow)
    plt.colorbar()  
    plt.show()
    
    return ax
