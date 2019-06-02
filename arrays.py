# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 2018
@name    Array Functions
@author: Jack Kriby Cook

"""

import xarray as xr
from collections import OrderedDict as ODict

from utilities.dispatchers import key_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['xarray_fromdata']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


@keydispatcher
def xarray_fromdata(datatype, data): raise KeyError(datatype)

@xarray_fromdata.register('dataframe')
def _xarray_fromdataframe(data, datakey): 
    uniquevalues = lambda column: list(set(data[column].values))
    scope = {column:uniquevalues(column)[0] for column in data.columns if len(uniquevalues(column)) == 1}
    headers = {column:uniquevalues(column) for column in data.columns if all([column not in scope, column != datakey])}
    xarray = xr.Dataset.from_dataframe(data[[datakey, *headers]].set_index(list(headers.keys()), drop=True))
    xarray.attrs = ODict([(key, value) for key, value in scope.items()])
    return xarray