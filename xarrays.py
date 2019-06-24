# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    XArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
import xarray as xr
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['xarray_fromdataframe', 'summation', 'mean', 'stdev', 'minimum', 'maximum',
           'normalize', 'standardize', 'minmax', 'scale', 'cumulate', 'interpolate']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def _uniquevalues(items): 
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]
            
            
# FACTORY
def xarray_fromdataframe(data, *args, key, **kwargs): 
    scope = ODict([(column, str(_uniquevalues(data[column])[0])) for column in data.columns if len(_uniquevalues(data[column])) == 1])
    headers = ODict([(column, [str(item) for item in _uniquevalues(data[column])]) for column in data.columns if all([column not in scope, column != key])])
    data = data[[key, *headers]].set_index(list(headers.keys()), drop=True).squeeze().to_xarray()    
    xarray = xr.DataArray(data.values, coords=headers, dims=list(headers.keys()), attrs=scope)
    return xarray


# REDUCTIONS
def summation(xarray, *args, axis, **kwargs): return xarray.sum(dim=axis, keep_attrs=True) 
def mean(xarray, *args, axis, **kwargs): return xarray.mean(dim=axis, keep_attrs=True)  
def stdev(xarray, *args, axis, **kwargs): return xarray.std(dim=axis, keep_attrs=True) 
def minimum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amin, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
def maximum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amax, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    

#def average(xarray, *args, axis, weights=None, **kwargs):
#    function = lambda x: np.average(x, axis=axis, weights=weights)
#    return xr.apply_ufunc(function, xarray, input_core_dims=[[axis]], keep_attrs=True)


# BROADCASTING
def normalize(xarray, *args, axis, **kwargs):
    xtotal = summation(xarray, *args, axis=axis, **kwargs)
    function = lambda x, t: np.divide(x, t)
    return xr.apply_ufunc(function, xarray, xtotal, keep_attrs=True)

def standardize(xarray, *args, axis, **kwargs):
    xmean = mean(xarray, *args, axis=axis, **kwargs)
    xstd = stdev(xarray, *args, axis=axis, **kwargs)
    function = lambda x, m, s: np.divide(np.subtract(x, m), s)
    return xr.apply_ufunc(function, xarray, xmean, xstd, keep_attrs=True)

def minmax(xarray, *args, axis, **kwargs):
    xmin = minimum(xarray, *args, axis=axis, **kwargs)
    xmax = maximum(xarray, *args, axis=axis, **kwargs)
    function = lambda x, i, a: np.divide(np.subtract(x, i), np.subtract(a, i))
    return xr.apply_ufunc(function, xarray, xmin, xmax, keep_attrs=True)
    
def scale(xarray, *args, method, axis, **kwargs):
    return {'normalize':normalize, 'standardize':standardize, 'minmax':minmax}[method](xarray, *args, axis=axis, **kwargs)    

def cumulate(xarray, *args, axis, direction='lower', **kwargs): 
    if direction == 'lower': return xarray.cumsum(dim=axis, keep_attrs=True)
    elif direction == 'upper': return xarray[{axis:slice(None, None, -1)}].cumsum(dim=axis, keep_attrs=True)[{axis:slice(None, None, -1)}]
    else: raise ValueError(direction)    
    
def interpolate(xarray, *args, values, axis, method='linear', fill='extrapolate', **kwargs):
    return xarray.interp(**{axis:values}, method=method)

#def inversion(xarray, *args, values, axis, method='linear', fill='extrapolate', **kwargs): 
#    pass









