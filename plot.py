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


def scatter_plot(coord_agg, fig=None):
    """
    Plots the cloud of points of the aggregate
    """
    # plot the cloud of points as a figure
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('y')
    ax.scatter(coord_agg[:, 0], coord_agg[:, 1],
               coord_agg[:, 2], marker='o', s=0.1)
    return fig


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


def bbox_plot(coord_agg, bbox, npoints_bbox=20, fig=None):
    """
    Plots the cloud of points of the aggregate and its bounding box
    Needs the coordinates of the aggregate (3D array),
    the optimized rotation angles between x and y and
    the number of points in each edge of the bounding box (npoints_bbox)
    """
    # plot the cloud of points as a figure
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.scatter(coord_agg[:, 0], coord_agg[:, 1],
               coord_agg[:, 2], marker='o', s=0.1)

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
    "simply dray an ellipsoid"

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
               coord_agg[:, 2], marker='o', s=0.1)

    # draw ellipsoid
    plot_ellipsoid(ellipsoid, ax=ax)

    return fig
