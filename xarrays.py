# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    XArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from functools import update_wrapper
from collections import OrderedDict as ODict

from utilities.dispatchers import keyword_singledispatcher as keyword_dispatcher
import utilities.narrays as nar

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_AGGREGATIONS = {'sum':np.sum, 'avg':np.mean, 'max':np.max, 'min':np.min}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_flatten = lambda nesteditems: [item for items in nesteditems for item in items]
_forceframe = lambda table: table.to_frame() if not isinstance(table, pd.DataFrame) else table


# FACTORY 
def xarray_fromdataframe(dataframe, *args, datakeys, attrs={}, aggs={}, forcedataset=True, **kwargs):
    assert all([key in dataframe.columns for key in datakeys])
    
    axeskeys = [key for key in dataframe.columns if key not in datakeys]
    axeskeys.sort(key=lambda key: len(set(dataframe[key].values)))

    dataframe = dataframe.set_index(axeskeys, drop=True)[datakeys[0] if len(datakeys) == 1 else list(datakeys)] 
    dataframe = _forceframe(dataframe)
    aggs = {key:_AGGREGATIONS[aggkey] for key, aggkey in aggs.items() if key in datakeys}
    if aggs: dataframe = dataframe.groupby(axeskeys).agg(aggs, axis=1)
    
    if len(dataframe.columns) == 1:
        dataarray = xr.DataArray.from_series(dataframe.squeeze()) 
        dataarray.name = datakeys[0]
        xarray = dataarray if not forcedataset else xr.Dataset({datakeys[0]:dataarray})
    elif len(dataframe.columns) > 1: 
        xarray =  xr.Dataset.from_dataframe(dataframe)
    else: raise ValueError(dataframe.columns)         
    xarray.attrs = attrs
    return xarray
    

def xarray_fromvalues(data, *args, axes, scope={}, attrs={}, forcedataset=True, **kwargs): 
    assert isinstance(data, ODict) and isinstance(axes, ODict)
    assert all([isinstance(item, np.ndarray) for item in data.values()])
    dataarrays = {key:xr.DataArray(values, coords=axes, dims=list(axes.keys()), name=key) for key, values in data.items()}
    if len(dataarrays) == 1 and not forcedataset: xarray = list(dataarrays.values())[0]
    else: xarray = xr.Dataset(dataarrays)
    xarray = xarray.assign_coords(**scope)
    xarray.attrs = attrs
    return xarray


# SUPPORT
def dataarray_function(function):
    def wrapper(dataarray, *args, **kwargs):
        assert isinstance(dataarray, xr.DataArray)
        newdataarray = function(dataarray, *args, **kwargs)
        newdataarray.attrs = dataarray.attrs
        newdataarray.name = dataarray.name
        return newdataarray
    update_wrapper(wrapper, function)
    return wrapper

@keyword_dispatcher('fill')
def fillcurve(*args, **kwargs): return {'bounds_error':True}
@fillcurve.register('extrapolate')
def extrapolate_fillcurve(*args, **kwargs): return {'fill_value':'extrapolate', 'bounds_error':False}
@fillcurve.register('bound')
def bounds_fillcurve(*args, bounds, **kwargs): return {'fill_value':bounds, 'bounds_error':False}


# REDUCTIONS
@dataarray_function
def summation(dataarray, *args, axis, **kwargs): return dataarray.sum(dim=axis, keep_attrs=True) 
@dataarray_function
def average(dataarray, *args, axis, **kwargs): return dataarray.mean(dim=axis, keep_attrs=True)  
@dataarray_function
def stdev(dataarray, *args, axis, **kwargs): return dataarray.std(dim=axis, keep_attrs=True) 

@dataarray_function
def minimum(dataarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amin, dataarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    
@dataarray_function
def maximum(dataarray, *args, axis, **kwargs): return xr.apply_ufunc(np.amax, dataarray, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1})    


@dataarray_function
def wtaverage(dataarray, *args, axis, weights, **kwargs): 
    function = lambda x: nar.wtaverage(x, index=-1, weights=weights)
    return xr.apply_ufunc(function, dataarray, input_core_dims=[[axis]], keep_attrs=True)  
@dataarray_function
def wtstdev(dataarray, *args, axis, weights, **kwargs): 
    function = lambda x: nar.wtstdev(x, index=-1, weights=weights)
    return xr.apply_ufunc(function, dataarray, input_core_dims=[[axis]], keep_attrs=True)  
@dataarray_function
def wtmedian(dataarray, *args, axis, weights, **kwargs): 
    function = lambda x: nar.wtmedian(x, index=-1, weights=weights)
    return xr.apply_ufunc(function, dataarray, input_core_dims=[[axis]], keep_attrs=True)  


# GROUPING
@dataarray_function
def groupby(dataarray, *args, axis, agg, axisgroups={}, **kwargs):
    if all([key == value[0] and len(value) == 1 for key, value in axisgroups.items()]): return dataarray
    function = lambda x, newvalue: xr.apply_ufunc(_AGGREGATIONS[agg], x, input_core_dims=[[axis]], keep_attrs=True, kwargs={'axis':-1}).assign_coords(**{axis:newvalue}).expand_dims(axis) 
    dataarrays = [dataarray.loc[{axis:_aslist(values)}] for values in axisgroups.values()] 
    dataarrays = [function(dataarray, newvalue) for dataarray, newvalue in zip(dataarrays, axisgroups.keys())]
    return xr.concat(dataarrays, axis)


# BROADCASTING
@dataarray_function
def normalize(dataarray, *args, axis=None, **kwargs):
    xtotal = summation(dataarray, *args, axis=axis, **kwargs)
    function = lambda x, t: np.divide(x, t)
    return xr.apply_ufunc(function, dataarray, xtotal, keep_attrs=True)

@dataarray_function
def standardize(dataarray, *args, axis=None, **kwargs):
    xmean = average(dataarray, *args, axis=axis, **kwargs)
    xstd = stdev(dataarray, *args, axis=axis, **kwargs)
    function = lambda x, m, s: np.divide(np.subtract(x, m), s)
    return xr.apply_ufunc(function, dataarray, xmean, xstd, keep_attrs=True)

@dataarray_function
def minmax(dataarray, *args, axis=None, **kwargs):
    xmin = kwargs.get('minimum', minimum(dataarray, *args, axis=axis, **kwargs))
    xmax = kwargs.get('maximum', maximum(dataarray, *args, axis=axis, **kwargs))
    function = lambda x, mi, ma: np.divide(np.subtract(x, mi), np.subtract(ma, mi))
    return xr.apply_ufunc(function, dataarray, xmin, xmax, keep_attrs=True) 

@dataarray_function
def absolute(dataarary, *args, **kwargs):
    return xr.apply_ufunc(np.abs, dataarary, keep_attrs=True)

@dataarray_function
def interpolate(dataarray, *args, values, axis, how, **kwargs):
    return dataarray.interp(**{axis:values}, method=how, kwargs=fillcurve(*args, **kwargs)) 


# ROLLING
@dataarray_function
def upper_cumulate(dataarray, *args, axis, **kwargs): 
    return lower_cumulate(dataarray[{axis:slice(None, None, -1)}], *args, axis=axis, **kwargs)[{axis:slice(None, None, -1)}]

@dataarray_function
def lower_cumulate(dataarray, *args, axis, **kwargs): 
    return dataarray.cumsum(dim=axis, keep_attrs=True)
 
@dataarray_function
def upper_uncumulate(dataarray, *args, axis, **kwargs): 
    return lower_uncumulate(dataarray[{axis:slice(None, None, -1)}], *args, axis=axis, **kwargs)[{axis:slice(None, None, -1)}]

@dataarray_function
def lower_uncumulate(dataarray, *args, axis,  total, **kwargs): 
    diffdataarray = moving_difference(dataarray, *args, axis=axis, period=1, **kwargs)
    residdataarray = total - summation(diffdataarray, *args, axis=axis, **kwargs)
    residdataarray = residdataarray.assign_coords({axis:dataarray.coords[axis].values[0]})
    return xr.concat([residdataarray, diffdataarray], dim=axis, data_vars='all')         

@dataarray_function
def moving_average(dataarray, *args, axis, period, **kwargs):
    assert isinstance(period, int)
    assert len(dataarray.coords[axis].values) >= period
    newdataarray = dataarray.rolling(**{axis:period+1}, center=True).mean().dropna(axis, how='all')
    return newdataarray

@dataarray_function
def moving_summation(dataarray, *args, axis, period, **kwargs):
    assert isinstance(period, int)
    assert len(dataarray.coords[axis].values) >= period
    newdataarray = dataarray.rolling(**{axis:period+1}, center=True).sum().dropna(axis, how='all')
    return newdataarray

@dataarray_function
def moving_difference(dataarray, *args, axis, period, **kwargs):
    assert isinstance(period, int)
    assert len(dataarray.coords[axis].values) >= period
    maxdataarray = dataarray.rolling(**{axis:period+1}, center=True).max().dropna(axis, how='all')
    mindataarray = dataarray.rolling(**{axis:period+1}, center=True).min().dropna(axis, how='all')
    return maxdataarray - mindataarray







