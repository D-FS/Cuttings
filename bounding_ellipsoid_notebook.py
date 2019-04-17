#!/usr/bin/env python
# coding: utf-8

import numpy as np
import test_image_ellipsoid as tie
import plot
import bounding_ellipsoid as be
import bounding_box as bb
import basic_functions as bf
import matplotlib.pyplot as plt

ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=10.,
                                     angles=(np.pi/7., np.pi/6.))
fig = plot.scatter_plot(aggregate)

bb.bbox_volume(aggregate)
bbox = bb.bbox_optim(aggregate)
print(bbox)
fig = plot.bbox_plot(aggregate, bbox)

rotated_aggregate = bf.rotate_aggregate(aggregate, angles=bbox['angles'])
rotated_bbox = bb.compute_bbox(rotated_aggregate)
fig = plot.bbox_plot(rotated_aggregate, rotated_bbox)

ellipsoid = be.bounding_ellipsoid_optim(aggregate)
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)

ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=30.,
                                     angles=(np.pi/7., np.pi/6.))
plot.scatter_plot(aggregate)

ellipsoid = be.bounding_ellipsoid_optim(aggregate, 1e-3)
rotated_ellipsoid = bf.rotate_aggregate(aggregate,
                                        angles=ellipsoid['bbox']['angles'])
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)

ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=1000.,
                                     angles=(np.pi/7., np.pi/6.))

plot.scatter_plot(aggregate)

ellipsoid = be.bounding_ellipsoid_optim(aggregate)
rotated_ellipsoid = bf.rotate_aggregate(aggregate,
                                        angles=ellipsoid['bbox']['angles'])
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)

plt.show()
