# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 2020
@name:   Utility Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
import numpy as np

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['UtilityIndex', 'UtilityFunction']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.sum(np.array(items))


UTILITY_FUNCTIONS = {
    'cobbdouglas': lambda a, d, s, w, x: (np.prod(np.power(np.subtract(x, s), w)) ** d) * a}
UTILITY_DERIVATIVES = {
    'cobbdouglas': lambda a, d, s, w, x, f, dx: ((w * d) / (x - s)) * f * dx}
INDEX_FUNCTIONS = {
    'additive': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), x)) * a,
    'inverted': lambda a, t, w, x: np.sum(np.divide(np.divide(w, t), x)) * a,
    'tangent': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))) * a,
    'rtangent': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))) * a,
    'logarithm': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.log(x + 1))) * a}


class UtilityIndex(ABC): 
    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass

    @property
    def key(self): 
        items = [(parameter, self.tolerances[parameter], self.weight[parameter]) for parameter in self.parameters] 
        return hash((self.functionname, self.functiontype, self.amplitude, tuple(items),))
    
    def __repr__(self): 
        string = '{}(functionname={}, functiontype={}, amplitude={}, tolerances={}, weights={})' 
        return string.format(self.__class__.__name__, self.functionname, self.functiontype, self.amplitude, self.tolerances, self.weights)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, tolerances={}, weights={}, **kwargs): 
        assert isinstance(tolerances, dict) and isinstance(weights, dict)
        assert all([hasattr(self, attr) for attr in ('functionname', 'functiontype', 'function', 'parameters')])
        self.amplitude = amplitude
        self.tolerances = {parm:tolerances.get(parm, 1) for parm in self.parameters}
        self.weights = {parm:weights.get(parm, 0) for parm in self.parameters}
 
    def __call__(self, *args, **kwargs): 
        values = self.execute(*args, **kwargs)
        assert isinstance(values, dict)
        assert all([parm in values.keys() for parm in self.parameters])
        t = np.array([self.tolerances[parm] for parm in self.parameters])
        w = np.array([self.weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([values[parm] for parm in self.parameters])
        return INDEX_FUNCTIONS[self.functiontype](self.amplitude, t, w, x)  

    @classmethod
    def register(cls, functionname, functiontype, *args, parameters, **kargs):
        if cls != UtilityIndex: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'register'))      
        assert isinstance(parameters, (tuple, list))
        assert functiontype in INDEX_FUNCTIONS.keys()
        attrs = dict(functionname=functionname, functiontype=functiontype, parameters=tuple(sorted(parameters)))
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper
  
    
class UtilityFunction(ABC): 
    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass
    def execute(self, *args, **kwargs): return {}

    @property
    def key(self): 
        items = [(parm, self.__subsistences[parm], self.__weights[parm]) for parm in self.parameters]   
        functions = [(parm, self.__functions[parm].key if parm in self.__functions.keys() else None) for parm in self.parameters]
        return hash((self.functionname, self.functiontype, self.__amplitude, tuple(items), tuple(functions)))
    
    def __repr__(self): 
        string = '{}(functionname={}, functiontype={}, amplitude={}, subsistences={}, weights={})' 
        return string.format(self.__class__.__name__, self.functionname, self.functiontype, self.__amplitude, self.__subsistences, self.__weights)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, diminishrate=1, subsistences={}, weights={}, functions={}, **kwargs): 
        assert isinstance(subsistences, dict) and isinstance(weights, dict)
        assert all([hasattr(self, attr) for attr in ('functionname', 'functiontype', 'parameters',)])
        self.__amplitude, self.__diminishrate = amplitude, diminishrate
        self.__subsistences = {parm:subsistences.get(parm, 0) for parm in self.parameters}
        self.__weights = {parm:weights.get(parm, 0) for parm in self.parameters}
        self.__functions = {parm:functions[parm] for parm in self.parameters if parm in functions}

    def __call__(self, *args, **kwargs):
        values = self.execute(*args, **kwargs)
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([self.__functions[parm](*args, **kwargs) if parm in self.__functions.keys() else values[parm] for parm in self.parameters])
        u = UTILITY_FUNCTIONS[self.functiontype](self.__amplitude, self.__diminishrate, s, w, x) if not np.any(x - s < 0) else np.NaN        
        return u

    def derivative(self, filtration, *args, **kwargs):
        values = self.execute(*args, **kwargs)
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([self.__functions[parm](*args, **kwargs) if parm in self.__functions.keys() else values[parm] for parm in self.parameters])
        u = UTILITY_FUNCTIONS[self.functiontype](self.__amplitude, self.__diminishrate, s, w, x)
        s, w, x = [i[self.parameters.index(_aslist(filtration[0]))] for i in (s, w, x)]
        dx = self.__functions[filtration[0]].derivative(filtration[1:], *args, **kwargs) if len(filtration) > 1 else 1
        du = UTILITY_DERIVATIVES[self.functiontype](self.__amplitude, self.__diminishrate, s, w, x, u, dx) if not np.any(x - s > 0) else np.NaN
        return du
        
    @classmethod
    def register(cls, functionname, functiontype, *args, parameters, **kwargs):
        if cls != UtilityFunction: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'register'))      
        assert isinstance(parameters, (tuple, list))
        assert functiontype in UTILITY_FUNCTIONS.keys()       
        attrs = dict(functionname=functionname, functiontype=functiontype, parameters=tuple(sorted(parameters)))
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper
    








        
        