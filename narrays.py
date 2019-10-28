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
__all__ = ['curve', 'inversion', 'interpolation', 'cumulate', 'uncumulate', 'movingaverage', 'movingtotal', 'wtaverage', 'wtstdev']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


#SUPPORT
@keyword_dispatcher('method')
def fillcurve(*args, **kwargs): return None

@fillcurve.register('extrapolate')
def extrapolate_fillcurve(*args, **kwargs): 
    return 'extrapolate'

@fillcurve.register('bound')
def bounds_fillcurve(*args, direction, bounds, **kwargs): 
    return {'upper': tuple(bounds[::-1]), 'lower': tuple(bounds[:])}[direction]


def curve(x, y, *args, how, fill={}, **kwargs): 
    fillvalue = fillcurve(*args, **fill, **kwargs)    
    return interp1d(x, y, kind=how, fill_value=fillvalue, bounds_error=False if fillvalue else True)
    

def wtaverage_vector(vector, weights):
    if all([np.isnan(x) for x in vector]): return np.nan    
    weights = [w for x, w in zip(vector, weights) if not np.isnan(x)]
    vector = [x for x in vector if not np.isnan(x)]
    assert len(vector) == len(weights)   
   
    weights = [w/sum(weights) for w in weights]
    vector = np.array([x * w for x, w in zip(vector, weights)])
    return np.sum(vector)

def wtstdev_vector(vector, weights):
    if all([np.isnan(x) for x in vector]): return np.nan    
    if len(vector) == 1: return 
    wtavg = wtaverage_vector(vector, weights)
    weights = [w for x, w in zip(vector, weights) if not np.isnan(x)]
    vector = [x for x in vector if not np.isnan(x)]
    assert len(vector) == len(weights)        
    
    if len(vector) == 1: return 0
    weights = [w/sum(weights) for w in weights]     
    vector = np.array([((x-wtavg)**2)*w for x, w in zip(vector, weights)])
    factor = ((len(vector)-1)/len(vector))
    return np.divide(sum(vector), factor)**0.5


def wtmedian_vector(vector, weights):
    if all([np.isnan(x) for x in vector]): return np.nan    
    weights = [w for x, w in zip(vector, weights) if not np.isnan(x)]
    vector = [x for x in vector if not np.isnan(x)]
    assert len(vector) == len(weights)      
    
    vector, weights = map(np.array, zip(*sorted(zip(vector, weights))))
    midpoint = 0.5 * sum(weights)
    if any(weights > midpoint): return (vector[weights == np.max(weights)])[0]
    else:
        weights = np.cumsum(weights)
        idx = np.where(weights <= midpoint)[0][-1]
        if weights[idx] == midpoint: return np.mean(vector[idx:idx+2])
        else: return vector[idx+1]


# BROADCASTING
def inversion(narray, header, values, *args, index, **kwargs):
    function = lambda x: curve(x, header, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)    

def interpolation(narray, header, values, *args, index, **kwargs):
    assert all([header[0] == item for item in header])
    function = lambda y: curve(header, y, *args, **kwargs)(values)
    return np.apply_along_axis(function, index, narray)


# REDUCTION
def wtaverage(narray, *args, index, weights, **kwargs): 
    return np.apply_along_axis(wtaverage_vector, index, narray, weights)

def wtstdev(narray, *args, index, weights, **kwargs): 
    return np.apply_along_axis(wtstdev_vector, index, narray, weights)

def wtmedian(narray, *args, index, weights, **kwargs): 
    return np.apply_along_axis(wtmedian_vector, index, narray, weights)


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












