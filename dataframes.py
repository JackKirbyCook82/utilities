# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   DataFrame Functions
@author: Jack Kirby Cook

"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['dataframe_fromjson', 'dataframe_fromhtml', 'dataframe_fromcsv', 'dataframe_fromfile', 'dataframe_tofile', 'dataframe_parser']
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

def dataframe_fromhtml(data, tablenum=0, header=None, htmlparser='lxml', forceframe=True):
    soup = bs(data, htmlparser)
    dataframe = pd.read_html(str(soup.find_all('table')), flavor=htmlparser, header=header)[tablenum]
    return _forceframe(dataframe) if forceframe else dataframe
    
def dataframe_fromcsv(data, header=None, forceframe=True):
    if data.endswith('\n'): data = data[:-2]
    data = [data.split(',') for data in data.split('\n')]        
    if header is None: dataframe = pd.DataFrame(data) 
    else: 
        cols = data.pop(header) 
        dataframe =  pd.DataFrame(data, columns=cols) 
    return _forceframe(dataframe) if forceframe else dataframe

def dataframe_fromxarray(data, key):
    series = data.to_series()
    series.name = key
    dataframe = series.to_frame().reset_index()
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


# CLEANERS
def dataframe_parser(dataframe, parsers={}, default=None):
    for column in dataframe.columns:
        try: dataframe.loc[:, column] = dataframe.loc[:, column].apply(parsers[column])
        except KeyError: 
            if default: dataframe.loc[:, column] = dataframe.loc[:, column].apply(default)
    return dataframe
  























