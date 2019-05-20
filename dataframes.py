# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   DataFrame Functions
@author: Jack Kirby Cook

"""

import os.path
import pandas as pd

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
  



