# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    Array Functions
@author: Jack Kriby Cook

"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d, interp2d
from collections import OrderedDict as ODict

from utilities.dispatchers import key_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['array_fromdata', 'interpolate1D', 'interpolate2D', 'normalize', 'standardize', 'minmax', 'scale', 'cumulate', 'average', 'mean']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def _uniquevalues(items): 
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


@keydispatcher
def array_fromdata(datatype, data): raise KeyError(datatype)

@array_fromdata.register('dataframe')
def _array_fromdataframe(data, datakey): 
    scope = ODict([(column, _uniquevalues(data[column])[0]) for column in data.columns if len(_uniquevalues(data[column])) == 1])
    headers = ODict([(column, _uniquevalues(data[column])) for column in data.columns if all([column not in scope, column != datakey])])
    data = data[[datakey, *headers]].set_index(list(headers.keys()), drop=True).squeeze().to_xarray()    
    xarray = xr.DataArray(data.values, coords=headers, dims=list(headers.keys()), attrs=scope)
    return xarray


def interpolate1D(array, header, values, *args, invert=False, kind='linear', fill='extrapolate', **kwargs): 
    if not invert: return interp1d(header, array, fill_value=fill, kind=kind)(values)
    else: return interp1d(array, header, fill_value=fill, kind=kind)(values) 

def interpolate2D(array, xheader, yheader, xvalues, yvalues, *args, fill='extrapolate', kind='linear', dx=0, dy=0, **kwargs):
    return interp2d(xheader, yheader, array, fill_value=fill, kind=kind)(xvalues, yvalues, dx=dx, dy=dy)


# BROADCASTING
def normalize(array, *args, **kwargs): return np.multiply(np.true_divide(array, np.sum(array)), 1)
def standardize(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.mean(array)) / np.std(array))(array)
def minmax(array, *args, **kwargs): return np.vectorize(lambda item: (item - np.amin(array))/(np.amax(array) - np.amin(array)))(array)
def scale(array, *args, scale, **kwargs): return {'normalize': normalize, 'standardize': standardize, 'minmax': minmax}(array)

def cumulate(array, *args, direction='lower', **kwargs): return {'lower': lambda x: np.cumsum(x), 'upper': lambda x: np.flip(np.cumsum(np.flip(x, 0)), 0)}[direction](array)


# REDUCTIONS
def average(array, *args, weights=None, **kwargs): return np.ma.average(array, weights=weights)
def mean(array, *args, **kwargs): return np.sum(array)






















