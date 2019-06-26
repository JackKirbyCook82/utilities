# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    NArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
from scipy.interpolate import interp1d

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['interpolate', 'cumulate']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


# BROADCASTING
def cumulate(narray, *args, axis, direction, **kwargs):
    function = {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction]
    return np.apply_along_axis(function, axis, narray)

def interpolate(narray, header, values, *args, axis, method, fill='extrapolate', invert=False, **kwargs):
    if not invert: function = lambda fx, x, vals: interp1d(x, fx, fill_value=fill, kind=method)(values)
    else: function = lambda fx, x, vals: interp1d(fx, x, fill_value=fill, kind=method)(values)
    return np.apply_along_axis(function, axis, narray, header, values)


