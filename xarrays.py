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
           'normalize', 'standardize', 'minmax', 'cumulate']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
def _uniquevalues(items): 
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]
            
            
# FACTORY
def xarray_fromdataframe(data, *args, datakey, headerkeys, scopekeys, **kwargs): 
    assert all([len(_uniquevalues(data[item])) == 1 for item in _aslist(scopekeys)])
    scope = ODict([(key, str(_uniquevalues(data[key])[0])) for key in _aslist(scopekeys)])
    headers = ODict([(key, [str(item) for item in _uniquevalues(data[key])]) for key in _aslist(headerkeys)])
    data = data[[datakey, *headers]].set_index(list(headers.keys()), drop=True).squeeze().to_xarray()    
    xarray = xr.DataArray(data, coords=headers, dims=list(headers.keys()), attrs=scope)
    return xarray

def xarray_fromvalues(data, *args, axes, scope, **kwargs): 
    return xr.DataArray(data, coords=axes, dims=list(axes.keys()), attrs=scope)


# REDUCTIONS
def summation(xarray, *args, axis, **kwargs): return xarray.sum(dim=axis, keep_attrs=True) 
def mean(xarray, *args, axis, **kwargs): return xarray.mean(dim=axis, keep_attrs=True)  
def stdev(xarray, *args, axis, **kwargs): return xarray.std(dim=axis, keep_attrs=True) 
def minimum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amin, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
def maximum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amax, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
def average(xarray, *args, axis, weights=None, **kwargs): return xarray.reduce(np.average, dim=axis, keep_attrs=None, weights=weights, **kwargs)


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

def cumulate(xarray, *args, axis, direction='lower', **kwargs): 
    if direction == 'lower': return xarray.cumsum(dim=axis, keep_attrs=True)
    elif direction == 'upper': return xarray[{axis:slice(None, None, -1)}].cumsum(dim=axis, keep_attrs=True)[{axis:slice(None, None, -1)}]
    else: raise ValueError(direction)    
    
def interpolate(xarray, *args, values, axis, method, fill='extrapolate', **kwargs):
    return xarray.interp(**{axis:values}, method=method)






