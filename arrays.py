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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['xarray_fromdataframe', 'apply_toxarray', 'apply_toarray', 'interpolate1D', 'interpolate2D', 
           'normalize', 'standardize', 'minmax', 'cumulate', 'average', 'summation', 'minimum', 'maximum']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def _uniquevalues(items): 
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


# FACTORY
def xarray_fromdataframe(data, key): 
    scope = ODict([(column, str(_uniquevalues(data[column])[0])) for column in data.columns if len(_uniquevalues(data[column])) == 1])
    headers = ODict([(column, [str(item) for item in _uniquevalues(data[column])]) for column in data.columns if all([column not in scope, column != key])])
    data = data[[key, *headers]].set_index(list(headers.keys()), drop=True).squeeze().to_xarray()    
    xarray = xr.DataArray(data.values, coords=headers, dims=list(headers.keys()), attrs=scope)
    return xarray


# APPLY PROTOCOAL
def apply_toxarray(xarray, function, *args, axis, **kwargs):
    ### FIX ###
    kwargs = {'axis':-1, **kwargs}
    parms = dict(input_core_dims=[[axis]], output_core_dims=[[axis]], vectorize=True)
    return xr.apply_ufunc(function, xarray, *args, **parms, keep_attrs=True, kwargs=kwargs)
    ### FIX ###

def apply_toarray(array, function, *args, axis, **kwargs):
    return np.apply_along_axis(function, axis, array, *args, **kwargs)


# BROADCASTING
def normalize(array, *args, **kwargs): return np.multiply(np.true_divide(array, np.sum(array)), 1)
def standardize(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.mean(array)) / np.std(array))(array)
def minmax(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.amin(array))/(np.amax(array) - np.amin(array)))(array)
def scale(array, *args, method, **kwargs): return {'normalize':normalize, 'standardize':standardize, 'minmax':minmax}[method](array, *args, **kwargs)

def cumulate(array, *args, direction='lower', **kwargs): 
    return {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction](array)

def interpolate1D(array, header, values, *args, invert=False, kind='linear', fill='extrapolate', **kwargs): 
    if not invert: return sp.interp1d(header, array, fill_value=fill, kind=kind)(values)
    else: return sp.interp1d(array, header, fill_value=fill, kind=kind)(values) 

def interpolate2D(array, xheader, yheader, xvalues, yvalues, *args, fill='extrapolate', kind='linear', dx=0, dy=0, **kwargs):
    return sp.interp2d(xheader, yheader, array, fill_value=fill, kind=kind)(xvalues, yvalues, dx=dx, dy=dy)


# REDUCTIONS
def average(array, *args, weights=None, **kwargs): return np.ma.average(array, weights=weights)
def summation(array, *args, **kwargs): return np.sum(array)
def minimum(array, *args, **kwargs): return np.amin(array)
def maximum(array, *args, **kwargs): return np.amax(array)



















