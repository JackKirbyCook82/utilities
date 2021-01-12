# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   DataFrame Functions
@author: Jack Kirby Cook

"""

import os
import pandas as pd
import xarray as xr
import geopandas as gp

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['dataframe_fromjson', 'dataframe_fromcsv', 'dataframe_fromxarray', 'dataframe_tofile', 'dataframe_fromfile', 'dataframe_parser']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_forceframe = lambda table: table.to_frame() if not isinstance(table, pd.DataFrame) else table


# FACTORY
def dataframe_fromjson(data, header=None, forceframe=True): 
    if header is None: dataframe = pd.DataFrame(data)
    else: 
        columns = data.pop(header)
        dataframe = pd.DataFrame(data, columns=columns)
    return _forceframe(dataframe) if forceframe else dataframe
    
def dataframe_fromcsv(data, header=None, forceframe=True):
    if data.endswith('\n'): data = data[:-2]
    data = [data.split(',') for data in data.split('\n')]        
    if header is None: dataframe = pd.DataFrame(data) 
    else: 
        cols = data.pop(header) 
        dataframe =  pd.DataFrame(data, columns=cols) 
    return _forceframe(dataframe) if forceframe else dataframe

def dataframe_fromxarray(data):
    if isinstance(data, xr.DataArray):
        series = data.to_series()
        series.name = data.name
        dataframe = series.to_frame().reset_index()      
    elif isinstance(data, xr.Dataset):
        dataframe = data.to_dataframe().reset_index()
    else: raise TypeError(type(data).__name__)
    for key, value in data.attrs.items(): dataframe[key] = value
    return dataframe


# FILE
def dataframe_tofile(file, dataframe, index=False, header=True): 
    assert str(file).endswith('.csv')
    directory, file = os.path.dirname(file), os.path.basename(file)
    try: filename, filecomp, fileext = str(file).split('.')
    except ValueError: filename, fileext = str(file).split('.')
    try: 
        compression = dict(method=filecomp, archive_name='.'.join([filename, fileext]))
        _forceframe(dataframe).to_csv(os.path.join(directory, '.'.join([filename, filecomp])), compression=compression, index=index, header=header)
    except NameError: _forceframe(dataframe).to_csv(os.path.join(directory, '.'.join([filename, fileext])), index=index, header=header)      
    
def dataframe_fromfile(file, index=None, header=0, forceframe=True):
    assert str(file).endswith('.csv')
    directory, file = os.path.dirname(file), os.path.basename(file)
    try: filename, filecomp, fileext = str(file).split('.')
    except ValueError: filename, fileext = str(file).split('.') 
    try: dataframe = pd.read_csv(os.path.join(directory, '.'.join([filename, filecomp])), compression=filecomp, index_col=index, header=header).dropna(axis=0, how='all')
    except NameError: dataframe = pd.read_csv(os.path.join(directory, '.'.join([filename, fileext])), index_col=index, header=header).dropna(axis=0, how='all')                
    return _forceframe(dataframe) if forceframe else dataframe    

def geodataframe_fromdir(directory):
    for file in os.listdir(directory):
        if file.endswith(".shp"): geodataframe = gp.read_file(os.path.join(directory, file))
    return geodataframe
        
def geodataframe_fromfile(file):
    geodataframe = gp.read_file(file)
    return geodataframe


# CLEANERS
def dataframe_parser(dataframe, parsers={}, defaultparser=None):
    for column in dataframe.columns:
        try: dataframe.loc[:, column] = dataframe.loc[:, column].apply(parsers[column])
        except KeyError: 
            if defaultparser:
                dataframe.loc[:, column] = dataframe.loc[:, column].apply(defaultparser)
    return dataframe
  























