# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 2020
@name:   Convergence Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['History', 'DurationDampener', 'ErrorConverger', 'ConvergenceError', 'OscillationConverger']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_multiply = lambda x, y: x * y
_divide = lambda x, y: x / y
_window = lambda x, n, i: x[i:i+n]
_void = lambda n: np.ones(n-1) * np.NaN
_pad = lambda x, n, f: np.concatenate([_void(n), f(x, n)])
_error = lambda x, rtol, atol: np.allclose(x, np.zeros(x.shape), rtol=rtol, atol=atol) 
_avgsqerr = lambda x: np.square(x).mean() ** 0.5
_maxsqerr = lambda x: np.max(np.square(x)) ** 0.5
_minsqerr = lambda x: np.min(np.square(x)) ** 0.5
_delta = lambda x: x[1:] - x[:-1]
_drt = lambda x: _delta(x) / np.abs(_delta(x))
#_mdelta = lambda x, n: np.convolve(_delta(x), np.ones(n), 'valid')
#_mdrt = lambda x, n: np.convolve(_drt(x), np.ones(n), 'valid')
_sma = lambda x, n: np.convolve(x, np.ones(n)/n, 'valid')
_mmax = lambda x, n: np.array([np.amax(_window(x, n, i)) for i in np.arange(len(x)-n+1)])
_mmin = lambda x, n: np.array([np.amin(_window(x, n, i)) for i in np.arange(len(x)-n+1)])
_ema = lambda x, n, s: (x[-1] * (s / (1 + n))) + (_ema(x[:-1], n, s) if len(x) > 1 else 0) * (1 - (s / (1 + n)))


class Dampener(ABC):
    @abstractmethod
    def execute(self, data): pass
    def __call__(self, data): return self.execute(data) 

class DurationDampener(Dampener): 
    def __init__(self, *args, period, size, **kwargs): 
        assert isinstance(period, int) and period > 0
        assert isinstance(size, float) and 0 < size < 1
        self.__period, self.__size = period, size
        
    def execute(self, data): 
        factors = np.floor((np.ones(data.shape[0]) * data.shape[-1]) / self.__period)
        sizes = np.ones(data.shape[0]) * self.__size
        return np.power(1 - sizes, factors)
    

class ConvergenceError(Exception): pass
class Converger(ABC): 
    @abstractmethod
    def converged(self): pass
    @abstractmethod
    def limit(self): pass
                
    def __bool__(self): return self.converged() if self.active else False
    def __len__(self): return self.values.shape[-1] if self.active else 0
    def __call__(self, errors, values): self.errors, self.values = errors, values     

    @property
    def active(self): return hasattr(self, 'values') and hasattr(self, 'errors')
    @property
    def value(self): 
        if not self or not self.active: raise ConvergenceError()
        else: return self.limit()
                
class ErrorConverger(Converger): 
    def __init__(self, *args, rtol, atol, **kwargs): self.__rtol, self.__atol = rtol, atol
    def limit(self): return self.values[:, -1]
    def converged(self): 
        if not self.active: return False
        else: return _error(self.errors, self.__rtol, self.__atol)

class OscillationConverger(Converger):
    def __init__(self, *args, period, tolerance, **kwargs): self.__period, self.__tolerance = period, tolerance
    def limit(self): return np.average(np.apply_along_axis(_sma, 1, self.values[:, -self.__period:], self.__period), axis=1)
    def converged(self): 
        if not self.active or len(self) <= self.__period: return False
        return all([self.__bounded(), self.__oscillating(), not self.__trending()])  

    def __bounded(self):
        xmax = np.apply_along_axis(np.max, 1, self.values[:, -self.__period:])
        xmin = np.apply_along_axis(np.min, 1, self.values[:, -self.__period:])
        return np.all(np.maximum((xmax - xmin) - self.__tolerance, 0) == 0)

#    def __trending(self): pass   
#    def __oscillating(self): pass

    
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
    
    
    
    
    
    
    
    
    
    
    