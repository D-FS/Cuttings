#!/usr/bin/env python3

import meshio
import akantu
import argparse
import os.path


def load_aggregate(filename):
    base, ext = os.path.splitext(filename)

    mesh = meshio.read(filename)
    meshio.write(base + '.vtk', mesh)
    return mesh
