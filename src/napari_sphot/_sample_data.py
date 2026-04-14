"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy
from skimage import io
import pandas as pd


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    scale = (30, 30, 30)
    units = ('nm', 'nm', 'nm')
    nuclei = io.imread('https://dev.mri.cnrs.fr/attachments/download/3983/02-SMALL_NUCLEI.tif')
    nucleiData = numpy.array(nuclei)
    spots = io.imread('https://dev.mri.cnrs.fr/attachments/download/3984/02-SMALL_SPOTS.tif')
    spotsData = numpy.array(spots)
    labels = io.imread('https://dev.mri.cnrs.fr/attachments/download/3982/Labels.tif')
    labelsData = numpy.array(labels)
    points = pd.read_csv('https://dev.mri.cnrs.fr/attachments/download/3981/Points.csv')
    pointsData = list(zip(points['axis-0'].values, points['axis-1'].values, points['axis-2'].values))

    return [(nucleiData,
             {'scale': scale,
              'name': 'nuclei',
              'units': units}),
            (spotsData,
             {'scale': scale,
              'name': 'spots',
              'units': units}),
            (labelsData,
             {'scale': scale,
              'name': 'labels',
              'units': units},
              "labels"),
            (pointsData,
             {
                 'scale': scale,
                 'name': 'points',
                 'units': units
             },
             "points"
            )
            ]
