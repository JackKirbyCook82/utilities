# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   DataFrame Functions
@author: Jack Kirby Cook

"""

import os.path
import pandas as pd
from bs4 import BeautifulSoup as bs

from utilities.dispatchers import key_singledispatcher as keydispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['dataframe_fromfile', 'dataframe_tofile', 'dataframe_parser']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_forceframe = lambda table: table.to_frame() if not isinstance(table, pd.DataFrame) else table


def dataframe_fromfile(file, index=None, header=0, forceframe=True):
    if not os.path.isfile(file): raise FileNotFoundError(file)
    dataframe = pd.read_csv(file, index_col=index, header=header).dropna(axis=0, how='all') 
    return _forceframe(dataframe) if forceframe else dataframe


def dataframe_tofile(file, dataframe, index=True, header=True): 
    return dataframe.to_csv(file, index=index, header=header)


def dataframe_parser(dataframe, parsers={}, default=None):
    for column in dataframe.columns:
        try: dataframe.loc[:, column] = dataframe.loc[:, column].apply(parsers[column])
        except KeyError: 
            if default: dataframe.loc[:, column] = dataframe.loc[:, column].apply(default)
    return dataframe
  

@keydispatcher
def dataframe_fromdata(datatype, data): raise KeyError(datatype)

@dataframe_fromdata.register('json')
def _dataframe_fromjsondata(data, header=None, forceframe=True): 
    if header is None: dataframe = pd.DataFrame(data)
    else: 
        columns = data.pop(header)
        dataframe = pd.DataFrame(data, columns=columns)
    return _forceframe(dataframe) if forceframe else dataframe

@dataframe_fromdata.register('html')
def _dataframe_fromhtmldata(data, tablenum=0, header=None, htmlparser='lxml', forceframe=True):
    soup = bs(data, htmlparser)
    dataframe = pd.read_html(str(soup.find_all('table')), flavor=htmlparser, header=header)[tablenum]
    return _forceframe(dataframe) if forceframe else dataframe
    
@dataframe_fromdata.register('csv')
def _dataframe_fromcsvdata(data, header=None, forceframe=True):
    if data.endswith('\n'): data = data[:-2]
    data = [data.split(',') for data in data.split('\n')]        
    if header is None: dataframe = pd.DataFrame(data) 
    else: 
        cols = data.pop(header) 
        dataframe =  pd.DataFrame(data, columns=cols) 
    return _forceframe(dataframe) if forceframe else dataframe






















