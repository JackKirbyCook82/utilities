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
__all__ = ['curve', 'inversion', 'interpolation', 'cumulate', 'uncumulate', 'movingaverage', 'movingtotal']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


#SUPPORT
@keyword_dispatcher('method')
def smoothcurve(x, y, *args, **kwargs): return x, y

@smoothcurve.register('cumulative')
def cumulative_smoothcurve(x, y, *args, direction, tolerance=5, **kwargs):
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

@fillcurve.register('bound')
def bounds_fillcurve(*args, direction, bounds, **kwargs): 
    return {'upper': tuple(bounds[::-1]), 'lower': tuple(bounds[:])}[direction]


def curve(x, y, *args, how, fill={}, smoothing={}, **kwargs): 
    x, y = smoothcurve(x, y, *args, **smoothing, **kwargs)
    fillvalue = fillcurve(*args, **fill, **kwargs)    
    return interp1d(x, y, kind=how, fill_value=fillvalue, bounds_error=False if fillvalue else True)


# BROADCASTING
def inversion(narray, header, values, *args, index, **kwargs):
    function = lambda x: curve(x, header, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)    

def interpolation(narray, header, values, *args, index, **kwargs):
    function = lambda y: curve(header, y, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)


# REDUCTION
def wtaverage(narray, *args, index, weights, **kwargs):
    ### INGORE NA
    pass

def wtstdev(narray, *args, index, weights, **kwargs):
    ### IGNORE NA
    pass


# ROLLING
def cumulate(narray, *args, index, direction, **kwargs):
    function = {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction]
    return np.apply_along_axis(function, index, narray)

def uncumulate(narray, *args, index, direction, **kwargs): 
    function = lambda x: [x[0]] + [x - y for x, y in zip(x[1:], x[:-1])]
    function = {'lower': lambda x: function(x), 'upper': lambda x: function(x[::-1])[::-1]}[direction]
    return np.apply_along_axis(function, index, narray)

def movingaverage(narray, *args, index, period, **kwargs):
    assert isinstance(period, int)
    assert narray.shape[index] >= period 
    function = lambda x: np.average(np.array([x[i:i+period] for i in range(0, len(x)-period+1)]), axis=1)    
    return np.apply_along_axis(function, index, narray)

def movingtotal(narray, *args, index, period, **kwargs):
    assert isinstance(period, int)
    assert narray.shape[index] >= period 
    function = lambda x: np.sum(np.array([x[i:i+period] for i in range(0, len(x)-period+1)]), axis=1)    
    return np.apply_along_axis(function, index, narray)












