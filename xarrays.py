# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    XArray Functions
@author: Jack Kriby Cook

"""

import numpy as np
import xarray as xr
from functools import update_wrapper
from collections import OrderedDict as ODict

import utilities.narrays as nar

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['xarray_fromdataframe', 'xarray_fromvalues', 'summation', 'average', 'stdev', 'minimum', 'maximum', 'wtaverage', 'wtstdev', 'wtmedian', 'groupby',
           'normalize', 'standardize', 'minmax', 'absolute', 'interpolate', 'lower_cumulate', 'upper_cumulate', 'lower_uncumulate', 'upper_uncumulate', 'moving_average', 'moving_summation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_AGGREGATIONS = {'sum':np.sum, 'avg':np.mean, 'max':np.max, 'min':np.min}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_flatten = lambda nesteditems: [item for items in nesteditems for item in items]


# FACTORY
def xarray_fromdataframe(data, *args, datakeys=[], datakey=None, aggs={}, fills={}, forcedataset=True, attrs={}, **kwargs):
    assert isinstance(attrs, dict)
    datakeys = [key for key in [*_aslist(datakey), *_aslist(datakeys)] if key]
    assert all([key in data.columns for key in datakeys])
    
    axeskeys = [key for key in data.columns if key not in datakeys]
    axeskeys.sort(key=lambda key: len(set(data[key].values)))
    for key in axeskeys: data.loc[:, key] = data[key].apply(str)

    data = data.set_index(axeskeys, drop=True)[datakeys] 
    aggs = {key:_AGGREGATIONS[aggkey] for key, aggkey in aggs.items() if key in datakeys}
    if aggs: data = data.groupby(axeskeys).agg(aggs, axis=1)

    if len(datakeys) == 1 and not forcedataset:
        dataarray = xr.DataArray.from_series(data).fillna(fills)   
        dataarray.name = datakeys[0]
        dataarray.attrs = attrs
        return dataarray
    else: 
        dataset = xr.Dataset.from_dataframe(data).fillna(fills)
        dataset.attrs = attrs
        return dataset
  

def xarray_fromvalues(data, *args, dims, scope, attrs, forcedataset=True, **kwargs): 
    assert all([isinstance(item, dict) for item in (data, attrs)])
    assert all([isinstance(dims, ODict) for item in (dims, scope)])
    assert all([key not in dims.keys() for key in scope.keys()])
    assert all([isinstance(items, np.ndarray) for items in data.values()])

    if len(data) == 1 and not forcedataset:
        dataarray = xr.DataArray(list(data.values())[0], coords=dims, dims=list(dims.keys()), attrs=attrs, name=list(data.keys())[0])
        dataarray = dataarray.assign_coords(**scope)
        return dataarray
    else: 
        dataset = xr.Dataset(data, coords=dims, dim=list(dims.keys()), attrs=attrs)
        dataset = dataset.assign_coords(**scope)
        return dataset


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
    axisgroups = {str(grpkey):[str(grpvalue) for grpvalue in grpvalues] for grpkey, grpvalues in axisgroups.items()}
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
    xmin = minimum(dataarray, *args, axis=axis, **kwargs)
    xmax = maximum(dataarray, *args, axis=axis, **kwargs)
    function = lambda x, mi, ma: np.divide(np.subtract(x, mi), np.subtract(ma, mi))
    return xr.apply_ufunc(function, dataarray, xmin, xmax, keep_attrs=True) 

@dataarray_function
def absolute(dataarary, *args, **kwargs):
    return xr.apply_ufunc(np.abs, dataarary, keep_attrs=True)

@dataarray_function
def interpolate(dataarray, *args, values, axis, how, fill, **kwargs):
    return dataarray.interp(**{axis:values}, how=how) 

# ROLLING
@dataarray_function
def upper_cumulate(dataarray, *args, axis, **kwargs): 
    return dataarray[{axis:slice(None, None, -1)}].cumsum(dim=axis, keep_attrs=True)[{axis:slice(None, None, -1)}] 

@dataarray_function
def lower_cumulate(dataarray, *args, axis, **kwargs): 
    return dataarray.cumsum(dim=axis, keep_attrs=True)
 
@dataarray_function
def upper_uncumulate(dataarray, *args, axis, **kwargs): 
    subfunction = lambda x: [x[0]] + [x - y for x, y in zip(x[1:], x[:-1])]
    function = lambda x: subfunction(x[::-1])[::-1]
    return xr.apply_ufunc(function, dataarray, input_core_dims=[[axis]], output_core_dims=[[axis]], keep_attrs=True)  

@dataarray_function
def lower_uncumulate(dataarray, *args, axis, **kwargs): 
    subfunction = lambda x: [x[0]] + [x - y for x, y in zip(x[1:], x[:-1])]
    function = lambda x: subfunction(x)
    return xr.apply_ufunc(function, dataarray, input_core_dims=[[axis]], output_core_dims=[[axis]], keep_attrs=True)

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

    








