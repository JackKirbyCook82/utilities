# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    NArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
from scipy.interpolate import interp1d

from utilities.dispatchers import keyword_singledispatcher as keyword_dispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['interpolation', 'inversion', 'cumulate']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


#SUPPORT
@keyword_dispatcher('method')
def smoothcurve(x, y, *args, **kwargs): return x, y

@smoothcurve.register('cumulative')
def cumulative_smoothcurve(x, y, *args, direction, tolerance=4, **kwargs):
    assert direction == 'lower' or direction == 'upper'
    assert len(x) == len(y)
    if direction == 'lower': x, y = x[::-1], y[::-1]
    dx = np.round(np.diff(x), decimals=tolerance)
    assert all([i <= 0 for i in dx])
    dxmap = [True] + [bool(i) for i in dx]
    function = lambda arr: np.array([val for i, val in zip(dxmap, arr) if i])                           
    return [function(i)[::-1] if direction == 'lower' else function(i) for i in (x, y)]


@keyword_dispatcher('method')
def fillcurve(*args, **kwargs): return None

@fillcurve.register('extrapolate')
def extrapolate_fillcurve(*args, **kwargs): 
    return 'extrapolate'

@fillcurve.register('bounds')
def bounds_fillcurve(*args, direction, boundarys, axis, **kwargs): 
    return {'upper': boundarys[axis][::-1], 'lower': boundarys[axis][:]}[direction]


def curve(x, y, *args, method, fill={}, smoothing={}, **kwargs): 
    x, y = smoothcurve(x, y, *args, **smoothing, **kwargs)
    fillvalue = fillcurve(*args, **fill, **kwargs)
    return interp1d(x, y, kind=method, fill_value=fillvalue, bounds_error=False if fillvalue else True)


# BROADCASTING
def inversion(narray, header, values, *args, index, **kwargs):
    function = lambda x: curve(x, header, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)    

def interpolation(narray, header, values, *args, index, **kwargs):
    function = lambda y: curve(header, y, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)

def cumulate(narray, *args, axis, direction, **kwargs):
    function = {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction]
    return np.apply_along_axis(function, axis, narray)

#def uncumulate(narray, *args, axis, direction, **kwargs): 
#   pass








