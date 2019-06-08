# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    Array Functions
@author: Jack Kriby Cook

"""

import numpy as np
import xarray as xr
import scipy as sp
from collections import OrderedDict as ODict

from utilities.dispatchers import key_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['xarray_fromdata', 'apply_toxarray', 'apply_toarray', 'interpolate1D', 'interpolate2D', 
           'normalize', 'standardize', 'minmax', 'scale', 'cumulate', 'average', 'summation']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def _uniquevalues(items): 
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


# FACTORY
@keydispatcher
def xarray_fromdata(datatype, data): raise KeyError(datatype)

@xarray_fromdata.register('dataframe')
def _xarray_fromdataframe(data, datakey): 
    scope = ODict([(column, _uniquevalues(data[column])[0]) for column in data.columns if len(_uniquevalues(data[column])) == 1])
    headers = ODict([(column, _uniquevalues(data[column])) for column in data.columns if all([column not in scope, column != datakey])])
    data = data[[datakey, *headers]].set_index(list(headers.keys()), drop=True).squeeze().to_xarray()    
    xarray = xr.DataArray(data.values, coords=headers, dims=list(headers.keys()), attrs=scope)
    return xarray

def apply_toxarray(xarray, function, *args, axis, **kwargs):
    kwargs={'axis':-1, **kwargs}
    return xr.apply_ufunc(function, xarray, *args, input_core_dims=[[axis]], output_core_dims=[[axis]], vectorize=True, keep_attrs=True, kwargs=kwargs)

def apply_toarray(array, function, *args, axis, **kwargs):
    return np.apply_along_axis(function, axis, array, *args, **kwargs)


# BROADCASTING
def normalize(array, *args, **kwargs): return np.multiply(np.true_divide(array, np.sum(array)), 1)
def standardize(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.mean(array)) / np.std(array))(array)
def minmax(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.amin(array))/(np.amax(array) - np.amin(array)))(array)
def scale(array, *args, scale, **kwargs): return {'normalize': normalize, 'standardize': standardize, 'minmax': minmax}(array)

def cumulate(array, *args, direction='upper', **kwargs): return {'upper': lambda x: np.cumsum(x), 'lower': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction](array)

def interpolate1D(array, header, values, *args, invert=False, kind='linear', fill='extrapolate', **kwargs): 
    if not invert: return sp.interp1d(header, array, fill_value=fill, kind=kind)(values)
    else: return sp.interp1d(array, header, fill_value=fill, kind=kind)(values) 

def interpolate2D(array, xheader, yheader, xvalues, yvalues, *args, fill='extrapolate', kind='linear', dx=0, dy=0, **kwargs):
    return sp.interp2d(xheader, yheader, array, fill_value=fill, kind=kind)(xvalues, yvalues, dx=dx, dy=dy)


# REDUCTIONS
def average(array, *args, weights=None, **kwargs): return np.ma.average(array, weights=weights)
def summation(array, *args, **kwargs): return np.sum(array)




















