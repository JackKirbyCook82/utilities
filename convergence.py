# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 2020
@name:   Convergence Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['History', 'MovingDampener', 'DeltaConverger']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_window = lambda x, n, i: x[i:i+n]
_void = lambda n: np.ones(n-1) * np.NaN
_pad = lambda x, n, f: np.concatenate([_void(n), f(x, n)])
_sma = lambda x, n: np.convolve(x, np.ones(n)/n, 'valid')
_mmax = lambda x, n: np.array([np.amax(_window(x, n, i)) for i in np.arange(len(x)-n+1)])
_mmin = lambda x, n: np.array([np.amin(_window(x, n, i)) for i in np.arange(len(x)-n+1)])
_avgsqerr = lambda x, y: np.round((np.square(x - y)).mean() ** 0.5, 3)
_maxsqerr = lambda x, y: np.round(np.max(np.square(x - y)) ** 0.5, 3)
_error = lambda x, rtol, atol: np.allclose(x, np.zeros(x.shape), rtol=rtol, atol=atol) 


class Converger(ABC): pass
class DeltaConverger(Converger):
    def __bool__(self): 
        try: return _error(self.__x - self.__y, self.__rtol, self.__atol)
        except AttributeError: return False
    def __init__(self, *args, rtol, atol, **kwargs): self.__rtol, self.__atol = rtol, atol 
    def __call__(self, x, y): self.__x, self.__y = x, y
    
    @property
    def avgerror(self): 
        try: return _avgsqerr(self.__x, self.__y)
        except: return None
    @property
    def maxerror(self): 
        try: return _maxsqerr(self.__x, self.__y)
        except: return None
    

class Dampener(ABC): pass
class MovingDampener(Dampener):
    def __init__(self, *args, period=25, size=1, minimum=0.01, **kwargs): 
        assert 0 < minimum < 1 and 0 < size <= 1
        self.__period, self.__size, self.__minimum = period, size, minimum
    
    def __call__(self, x, *args, **kwargs): 
        if len(x) <= self.__period: return 1
        y = x - np.concatenate([_void(self.__period), _sma(x, self.__period)])
        z = y / abs(y)
        z = z[~np.isnan(z)]        
        assert all(abs(z) == 1) if len(z) > 0 else True
        try: dampener = 1 - (np.sum(z[1:] * z[:-1] < 0) / (len(z) - 1))
        except ZeroDivisionError: return 1
        return min([max([dampener * self.__size, self.__minimum]), 1]) 
            
    
class History(object):
    @property
    def data(self): return self.__data
    def sma(self, period=1): return np.apply_along_axis(_sma, 1, self.__data) if len(self) >= period else _void(len(self))
    def mmax(self, period=1): return np.apply_along_axis(_mmax, 1, self.__data) if len(self) >= period else _void(len(self))
    def mmin(self, period=1): return np.apply_along_axis(_mmin, 1, self.__data) if len(self) >= period else _void(len(self))
  
    def __len__(self): return len(self.__data) if self else 0
    def __bool__(self): return hasattr(self, '__data')
    def __call__(self, data):
        try: self.__data = np.append(self.__data, np.expand_dims(data, axis=1), axis=1)
        except AttributeError: self.__data = np.expand_dims(data, axis=1)     

    def __getitem__(self, index):
        def wrapper(period=1):
            assert isinstance(period, int) and period >= 1
            columns = ['DATA', 'SMA{}'.format(period), 'MAX{}'.format(period), 'MIN{}'.format(period)]
            data = np.array([self.__data[index, :], _pad(self.__data[index, :], period, _sma), _pad(self.__data[index, :], period, _mmax), _pad(self.__data[index, :], period, _mmin)])
            dataframe = pd.DataFrame(data.transpose(), columns=columns)
            return dataframe
        return wrapper

    def table(self, period=1):
        assert isinstance(period, int) and period >= 1
        dataframe = pd.DataFrame(self.__data.transpose())
        if period > 0: dataframe = dataframe.rolling(window=period).mean().dropna(axis=1, how='all')
        return dataframe
    
    
    
    
    
    
    
    
    
    
    