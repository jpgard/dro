"""
Utilities for testing.
"""

import os

import numpy as np


def assert_shape_equal(arr1, arr2):
    assert arr1.shape == arr2.shape, "expected identical shapes, got shapes {} {}".format(
        arr1.shape, arr2.shape
    )
    return


def assert_ndims(arr: np.ndarray, ndims: int):
    assert len(arr.shape) == ndims, "expected array of shape {}; got {}".format(
        ndims, len(arr.shape)
    )
    return


def assert_file_exists(fp):
    assert os.path.exists(fp), "file {} does not exist.".format(fp)
    return
