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
__all__ = []
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_replace = lambda items, index, replacement: [replacement if i == index else item for i, item in enumerate(items)]
_bins = lambda x, grps: np.digitize(x, grps)
_histogram = lambda x: np.unique(x, return_counts=True)
_mask = lambda x: np.ma.masked_array(x, np.isnan(x))
_normalize = lambda x: x / np.nansum(x)
_minmax = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
_summation = lambda x: np.nansum(x) 
_average = lambda x, w: np.ma.average(_mask(x), weights=w).filled(np.nan)

#SUPPORT
@keyword_dispatcher('fill')
def _fillcurve(*args, **kwargs): 
    return {'bounds_error':True}

@_fillcurve.register('extrapolate')
def _extrapolate(*args, **kwargs): 
    return {'fill_value':'extrapolate', 'bounds_error':False}

@_fillcurve.register('bound')
def _bounds(*args, direction, bounds, **kwargs): 
    bounds = {'upper':tuple(bounds[::-1]), 'lower':tuple(bounds[:])}[direction]
    return {'fill_value':bounds, 'bounds_error':False}

def _curve(x, y, *args, how, **kwargs):  
    return interp1d(x, y, kind=how, **_fillcurve(*args, **kwargs))
    
def _wtaverage(x, w):    
    assert isinstance(x, np.ndarray) and isinstance(w, np.ndarray) and len(x) == len(w)
    if all([np.isnan(i) for i in x]): return np.nan    
    else: return _average(x, w)

def _distribution(size, values, function):
    groups = {key:value for key, value in zip(*_histogram(_bins(function(size), values)))}
    return [groups.get(i, 0) for i in range(len(values)+1)]   


# BROADCASTING
def inversion(narray, header, values, *args, index, **kwargs):
    function = lambda x: _curve(x, header, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)    

def interpolation(narray, header, values, *args, index, **kwargs):
    assert all([header[0] == item for item in header])
    function = lambda y: _curve(header, y, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)


# REDUCTION
def wtaverage(narray, *args, index, weights, **kwargs): 
    return np.apply_along_axis(_wtaverage, index, narray, weights)


# EXPANSION
def distribution(narray, *args, index, values, function, **kwargs):
    assert isinstance(values, (tuple, list))
    assert narray.shape[index] == 1
    return np.apply_along_axis(_distribution, index, narray, values, function)

def equaldivision(narray, *args, index, values, **kwargs):
    assert isinstance(values, (tuple, list))
    assert narray.shape[index] == len(values)
    items = np.array_split(narray, narray.shape[index], axis=index)
    items = [np.broadcast_to(item, _replace(narray.shape, index, value)) / value for item, value in zip(items, values)]
    return np.concatenate(items, axis=index)

def equalbroadcast(narray, *args, index, values, **kwargs):
    assert isinstance(values, (tuple, list))
    assert narray.shape[index] == len(values)    
    items = np.split(narray, narray.shape[index], axis=index)
    items = [np.broadcast_to(item, _replace(narray.shape, index, value)) for item, value in zip(items, values)]
    return np.concatenate(items, axis=index)
    

# ROLLING
def cumulate(narray, *args, index, direction, **kwargs):
    function = {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction]
    return np.apply_along_axis(function, index, narray)

def uncumulate(narray, *args, index, direction, **kwargs): 
    function = lambda x: np.convolve(x, np.array([-1, 1], 'valid'))
    function = {'lower': lambda x: function(x), 'upper': lambda x: function(x[::-1])[::-1]}[direction]    
    return np.apply_along_axis(function, index, narray)

def movingaverage(narray, *args, index, period, **kwargs):
    assert isinstance(period, int)
    assert narray.shape[index] >= period 
    function = lambda x: np.convolve(x, np.ones(period)/period, 'valid')   
    return np.apply_along_axis(function, index, narray)

def movingtotal(narray, *args, index, period, **kwargs):
    assert isinstance(period, int)
    assert narray.shape[index] >= period 
    function = lambda x: np.convolve(x, np.ones(period), 'valid')
    return np.apply_along_axis(function, index, narray)












