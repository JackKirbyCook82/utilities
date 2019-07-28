# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    XArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
import xarray as xr
from collections import OrderedDict as ODict
from functools import update_wrapper

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_aggregations = {'sum':np.sum, 'avg':np.mean, 'max':np.max, 'min':np.min}


# FACTORY
def xarray_fromdataframe(data, *args, datakeys, axekeys, attrkeys, aggs={}, fills={}, forcedataset=True, **kwargs):
    assert all([isinstance(item, (str, tuple, list)) for item in (datakeys, axekeys, attrkeys)])
    datakeys, axekeys, attrkeys = [_aslist(item) for item in (datakeys, axekeys, attrkeys)]    
    assert all([key in data.columns for key in (*datakeys, *axekeys, *attrkeys)])        
    assert all([len(set(data[key].values)) == 1 for key in attrkeys if key in data.columns])
    
    assert all([isinstance(item, dict) for item in (aggs, fills)])
    fills = {key:value for key, value in fills.items() if key in datakeys}
    aggs = {key:(_aggregations.get(value, value) if isinstance(value, str) else value) for key, value in aggs.items() if key in datakeys}
    
    for axis in axekeys: data.loc[:, axis] = data[axis].apply(str)
    for key in attrkeys: data.loc[:, key] = data[key].apply(str)
    
    attrs = ODict([(key, data[key].unique()) for key in attrkeys])
    assert all([len(value) == 1 for value in attrs.values()])
    attrs = ODict([(key, value[0]) for key, value in attrs.items()])   

    data = data.set_index(axekeys, drop=True)[datakeys]    
    if aggs: data = data.groupby(axekeys).agg(aggs)

    if len(datakeys) == 1 and not forcedataset:
        xarray = xr.DataArray.from_series(data).fillna(fills)     
        xarray.name = datakeys[0]
    else: xarray = xr.Dataset.from_dataframe(data).fillna(fills)
    xarray.attrs = attrs
    return xarray

def xarray_fromvalues(data, *args, axes, scope, forcedataset=True, **kwargs): 
    assert all([isinstance(item, dict) for item in (data, axes, scope)])
    assert all([isinstance(items, np.ndarray) for items in data.values()])
    if len(data) == 1 and not forcedataset:
        xarray = xr.DataArray(list(data.values())[0], coords=axes, dims=list(axes.keys()), attrs=scope)
        xarray.name = list(data.keys())[0]  
    else: xarray = xr.Dataset(data, coords=axes, dim=list(axes.keys()), attrs=scope)
    return xarray


# SUPPORT
def xarray_keepattrs(function):
    def wrapper(xarray, *args, **kwargs):
        newxarray = function(xarray, *args, **kwargs)
        newxarray.attrs = xarray.attrs
        if isinstance(newxarray, xr.DataArray): newxarray.name = xarray.name
        return newxarray
    update_wrapper(wrapper, function)
    return wrapper


# REDUCTIONS
@xarray_keepattrs
def summation(xarray, *args, axis, **kwargs): return xarray.sum(dim=axis, keep_attrs=True) 
@xarray_keepattrs
def mean(xarray, *args, axis, **kwargs): return xarray.mean(dim=axis, keep_attrs=True)  
@xarray_keepattrs
def stdev(xarray, *args, axis, **kwargs): return xarray.std(dim=axis, keep_attrs=True) 
@xarray_keepattrs
def minimum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amin, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
@xarray_keepattrs
def maximum(xarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amax, xarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
@xarray_keepattrs
def average(xarray, *args, axis, weights, **kwargs): return xarray.reduce(np.average, dim=axis, keep_attrs=None, weights=weights, **kwargs)
@xarray_keepattrs
def weightaverage(xarray, *args, axis, weights=None, **kwargs): return xarray.reduce(np.average, dim=axis, keep_attrs=None, weights=weights, **kwargs)


# BROADCASTING
@xarray_keepattrs
def normalize(xarray, *args, axis, **kwargs):
    xtotal = summation(xarray, *args, axis=axis, **kwargs)
    function = lambda x, t: np.divide(x, t)
    return xr.apply_ufunc(function, xarray, xtotal, keep_attrs=True)

@xarray_keepattrs
def standardize(xarray, *args, axis, **kwargs):
    xmean = mean(xarray, *args, axis=axis, **kwargs)
    xstd = stdev(xarray, *args, axis=axis, **kwargs)
    function = lambda x, m, s: np.divide(np.subtract(x, m), s)
    return xr.apply_ufunc(function, xarray, xmean, xstd, keep_attrs=True)

@xarray_keepattrs
def minmax(xarray, *args, axis, **kwargs):
    xmin = minimum(xarray, *args, axis=axis, **kwargs)
    xmax = maximum(xarray, *args, axis=axis, **kwargs)
    function = lambda x, i, a: np.divide(np.subtract(x, i), np.subtract(a, i))
    return xr.apply_ufunc(function, xarray, xmin, xmax, keep_attrs=True) 

@xarray_keepattrs
def interpolate(xarray, *args, values, axis, how, fill, **kwargs):
    return xarray.interp(**{axis:values}, how=how)

# ROLLING
@xarray_keepattrs
def cumulate(xarray, *args, axis, direction, **kwargs): 
    if direction == 'lower': return xarray.cumsum(dim=axis, keep_attrs=True)
    elif direction == 'upper': return xarray[{axis:slice(None, None, -1)}].cumsum(dim=axis, keep_attrs=True)[{axis:slice(None, None, -1)}]
    else: raise ValueError(direction)    

@xarray_keepattrs
def uncumulate(xarray, *args, axis, direction, **kwargs): 
    subfunction = lambda x: [x[0]] + [x - y for x, y in zip(x[1:], x[:-1])]
    function = {'lower': lambda x: subfunction(x), 'upper': lambda x: subfunction(x[::-1])[::-1]}[direction]
    return xr.apply_ufunc(function, xarray, input_core_dims=[[axis]], output_core_dims=[[axis]], keep_attrs=True)  

@xarray_keepattrs
def movingaverage(xarray, *args, axis, period, **kwargs):
    assert isinstance(period, int)
    assert len(xarray.coords[axis].values) >= period
    newxarray = xarray.rolling(**{axis:period+1}, center=True).mean().dropna(axis)
    newxarray.attrs = xarray.attrs
    return newxarray

@xarray_keepattrs
def movingtotal(xarray, *args, axis, period, **kwargs):
    assert isinstance(period, int)
    assert len(xarray.coords[axis].values) >= period
    newxarray = xarray.rolling(**{axis:period+1}, center=True).sum().dropna(axis)
    newxarray.attrs = xarray.attrs
    return newxarray

    








