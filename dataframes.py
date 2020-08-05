# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   DataFrame Functions
@author: Jack Kirby Cook

"""

import os
import pandas as pd
import numpy as np
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
    else: raise TypeError(data)
    for key, value in data.attrs.items(): dataframe[key] = value
    return dataframe


# FILE
def dataframe_tofile(file, dataframe, index=True, header=True): 
    try: 
        dataframe = dataframe.replace(np.nan, '', regex=True)
        dataframe.to_csv(file, index=index, header=header)
        print('File Saving Success:')
        print(str(file), '\n')                
    except Exception as error:
        print('File Saving Failure:')
        print(str(file), '\n')                  
        raise error     
    
def dataframe_fromfile(file, index=None, header=0, forceframe=True):
    try:         
        dataframe = pd.read_csv(file, index_col=index, header=header).dropna(axis=0, how='all')
        print('File Loading Success:')
        print(str(file), '\n')                
    except Exception as error:
        print('File Loading Failure:')
        print(str(file), '\n')                  
        raise error 
    return _forceframe(dataframe) if forceframe else dataframe    

def geodataframe_fromdir(directory):
    try:
        for file in os.listdir(directory):
            if file.endswith(".shp"): geodataframe = gp.read_file(os.path.join(directory, file))
        print('File Loading Success:')
        print(str(directory), '\n')   
    except Exception as error:
        print('File Loading Failure:')
        print(str(directory), '\n')         
        raise error
    return geodataframe
        
def geodataframe_fromfile(file):
    try:
        geodataframe = gp.read_file(file)
        print('File Loading Success:')
        print(str(file), '\n')   
    except Exception as error:
        print('File Loading Failure:')
        print(str(file), '\n')         
        raise error
    return geodataframe


# CLEANERS
def dataframe_parser(dataframe, parsers={}, defaultparser=None):
    for column in dataframe.columns:
        try: dataframe.loc[:, column] = dataframe.loc[:, column].apply(parsers[column])
        except KeyError: 
            if defaultparser:
                dataframe.loc[:, column] = dataframe.loc[:, column].apply(defaultparser)
    return dataframe
  























