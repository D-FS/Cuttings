#!/usr/bin/env python
# coding: utf-8

# # Bounding ellipsoid notebook

# In[1]:


import numpy as np
import test_image_ellipsoid as tie
import plot
import bounding_ellipsoid as be
import bounding_box as bb
import matplotlib.pyplot as plt
import basic_functions as bf

# ## Test with small noise

# Creation of an ellipsoidal test image with a = 100, b = 50, c = 30 and a noise amplitude of 10.   
# The produced cloud of points is rotated by an angle of pi/7 around the x axis et pi/6 around the y axis

# In[2]:


ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=10.,
                                     angles=(np.pi/7., np.pi/6.))
fig = plot.scatter_plot(aggregate)


# We can find the optimal bounding box

# In[4]:


bb.bbox_volume(aggregate)
bbox = bb.bbox_optim(aggregate)
print(bbox)
fig = plot.bbox_plot(aggregate, bbox)

# In[5]:

rotated_aggregate = bf.rotate_aggregate(aggregate, angles=bbox['angles'])
rotated_bbox = bb.compute_bbox(rotated_aggregate)
fig = plot.bbox_plot(rotated_aggregate, rotated_bbox)

ellipsoid = be.bounding_ellipsoid_optim(aggregate)
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)


# ## Test with big noise

# Creation of an ellipsoidal test image with a = 100, b = 50, c = 30 and a noise amplitude of 30.   
# The produced cloud of points is rotated by an angle of pi/7 around the x axis et pi/6 around the y axis

# In[4]:


ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=30.,
                                     angles=(np.pi/7., np.pi/6.))
plot.scatter_plot(aggregate)


# In[17]:


ellipsoid = be.bounding_ellipsoid_optim(aggregate, 1e-3)
rotated_ellipsoid = bf.rotate_aggregate(aggregate,
                                        angles=ellipsoid['bbox']['angles'])
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)

# ## Test with ultra-big noise

# Creation of an ellipsoidal test image with a = 100, b = 50, c = 30 and a noise amplitude of 1000.   
# The produced cloud of points is rotated by an angle of pi/7 around the x axis et pi/6 around the y axis

# In[18]:


ellipsoid = {'a': 100, 'b': 50, 'c': 30}
aggregate = tie.ellipsoid_test_image(ellipsoid,
                                     npoints=10000,
                                     noise_amplitude=1000.,
                                     angles=(np.pi/7., np.pi/6.))

plot.scatter_plot(aggregate)

# In[19]:


ellipsoid = be.bounding_ellipsoid_optim(aggregate)
rotated_ellipsoid = bf.rotate_aggregate(aggregate,
                                        angles=ellipsoid['bbox']['angles'])
plot.fit_ellipsoid_plot(rotated_aggregate, ellipsoid)
