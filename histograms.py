# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 2020
@name:   Histogram Objects
@author: Jack Kirby Cook

"""

import numpy as np
from scipy import stats

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Category_Histogram', 'Numeric_Histogram']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_NAMEFORMAT = '{}: "{}"'
_VALUESFORMAT = 'VALUES = {}: \n{}'
_WEIGHTSFORMAT = 'WEIGHTS = {}: \n{}'

_normalize = lambda x: x / np.sum(x)
_flatten = lambda nesteditems: [item for items in nesteditems for item in items]


def partition(values, weights, Variable, *args, cuts=0, **kwargs):
    applybounds = lambda x: [kwargs['bounds'][i] if x[i] is None else x[0] for i in range(len(x))]
    rangevalues = [applybounds(Variable.fromstr(value).value) for value in values]
    values = [np.mean(rangevalue) for rangevalue in rangevalues]
    for cut in range(cuts):
        rangevalues = _flatten([([rangevalue[0], value], [value, rangevalue[-1]]) for value, rangevalue in zip(values, rangevalues)])
        values = [np.mean(rangevalue) for rangevalue in rangevalues]
    weights = _flatten([[weight/cuts]*cuts for weight in weights]) if cuts > 0 else weights
    return values, weights


class Histogram(object):
    def sample(self, size): return self.__historgram.rvs(size=(1, size))
    
    @property
    def name(self): '|'.join([self.__weightname, self.__valuename])         
    @property
    def values(self): return self.__values
    @property
    def weights(self): return self.__weights
    @property
    def histogram(self): return self.__histogram
    
    def __repr__(self): 
        fmt = '{}({valuename={}, weightname={}, value={}, weights=[]})'
        return fmt.format(self.__class__.__name__, ', '.join([self.__valuename, self.__weightname, self.__values, self.__weights]))  
    
    def __str__(self): 
        namestr = _NAMEFORMAT.format(self.__class__.__name__, self.name)
        valuestr = _VALUESFORMAT.format(self.__valuename, self.__values)
        weightstr = _WEIGHTSFORMAT.format(self.__weightname, self.__weights)
        return '\n'.join([namestr, valuestr, weightstr])       

    def __init__(self, valuename, weightname, values, weights): 
        assert isinstance(values, list) and isinstance(weights, list)
        assert len(values) == len(weights)
        self.__values, self.__weights = np.array(values), np.array(weights)
        self.__valuename, self.__weightname = valuename, weightname
        self.__histogram = stats.rv_discrete(name=self.name, values=(self.__values, _normalize(self.__weights)))

    @classmethod
    def fromArrayTable(cls, arraytable, *args, **kwargs):
        assert arraytable.layers == 1 and arraytable.dims == 1
        valuename = arraytable.headerkeys[0]
        if arraytable.variables[valuename].datatype != 'num': return Numeric_Histogram.fromArrayTable(arraytable, *args, **kwargs)
        elif arraytable.variables[valuename].datatype != 'range': return Numeric_Histogram.fromArrayTable(arraytable, *args, **kwargs)
        elif arraytable.variables[valuename].datatype != 'category': return Category_Histogram.fromArrayTable(arraytable, *args, **kwargs)   
        else: raise ValueError(arraytable.variables[valuename].datatype)
            

class Category_Histogram(object):
    @classmethod
    def fromArrayTable(cls, arraytable, *args, **kwargs):
        assert arraytable.layers == 1 and arraytable.dims == 1
        valuename, weightname = arraytable.headerkeys[0], arraytable.datakeys[0]
        values, weights = arraytable.headers[valuename], arraytable.arrays[weightname] 
        if arraytable.variables[valuename].datatype != 'category': raise ValueError(arraytable.variables[valuename].datatype)
        return cls(valuename, weightname, values, weights)    
    

class Numeric_Histogram(object):
    @property
    def array(self): return np.array([np.full(weight, value) for value, weight in zip(np.nditer(self.values, np.nditer(self.weights)))]).flatten()        \
    
    def mean(self): return self.histogram.mean()
    def median(self): return self.histogram.median()
    def std(self): return self.histogram.std()
    def rstd(self): return self.std() / self.mean()
    def skew(self): return stats.skew(self.array)
    def kurtosis(self): return stats.kurtosis(self.array)
    
    @classmethod
    def fromArrayTable(cls, arraytable, *args, **kwargs):
        assert arraytable.layers == 1 and arraytable.dims == 1
        valuename, weightname = arraytable.headerkeys[0], arraytable.datakeys[0]
        values, weights = arraytable.headers[valuename], arraytable.arrays[weightname]
        Variable = arraytable.variables[valuename]
        if Variable.datatype == 'num': values = [Variable.fromstr(value).value for value in values]
        elif Variable.datatype == 'range': values, weights = partition(values, weights, Variable, *args, **kwargs)
        else: raise ValueError(Variable.datatype)
        return cls(valuename, weightname, values, weights)
        