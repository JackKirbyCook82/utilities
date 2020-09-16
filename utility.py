# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 2020
@name:   Utility Objects
@author: Jack Kirby Cook

"""

import numpy as np
import warnings
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['UtilityIndex', 'UtilityFunction']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.sum(np.array(items))


UTILITY_FUNCTIONS = {
    'cobbdouglas': lambda x, w, a, d, *args: (np.prod(np.power(x, w)) ** d) * a,
    'ces': lambda x, w, a, d, p, *args: (np.sum(np.multiply(np.power(x, p), w)) ** (d/p)) * a,
    'linear': lambda x, w, a, *args: np.sum(np.multiply(x, w)) ** a,
    'expcobbdouglas': lambda x, w, a, d, *args: (np.prod(np.power(np.exp(x), w)) ** d) * a}
UTILITY_DERIVATIVES = {
    'cobbdouglas': lambda i, x, w, a, d, *args: a * w[i] * d * (1/x[i]) * (np.prod(np.power(x, w)) ** d),
    'ces': lambda i, x, w, a, d, p, *args: a * w[i] * d * (x[i]**(p-1)) * (np.sum(np.multiply(x, w) ** p) ** ((d/p)-1)),
    'linear': lambda i, x, w, a, d, *args: a * w[i],
    'expcobbdouglas': lambda i, x, w, a, d, *args: a * w[i] * d * (np.prod(np.power(np.exp(x), w)) ** d)}
INDEX_FUNCTIONS = {
    'additive': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), x)) * a,
    'inverted': lambda x, w, t, a: np.sum(np.divide(np.divide(w, t), x)) * a,
    'tangent': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))) * a,
    'rtangent': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))) * a,
    'logarithm': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.log(x + 1))) * a}


class NumericalError(Exception): pass


class UtilityIndex(ABC): 
    def __init_subclass__(cls, functionname, functiontype, *args, parameters=[], coefficents=[], **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(parameters, (tuple, list)) and isinstance(coefficents, (tuple, list))
        assert functiontype in INDEX_FUNCTIONS.keys()
        setattr(cls, 'funcitonname', functionname)
        setattr(cls, 'functiontype', functiontype)
        setattr(cls, 'parameters', tuple(sorted(parameters)))
        setattr(cls, 'coefficents', tuple(coefficents))      
   
    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass    
    @abstractmethod
    def execute(self, *args, **kwargs): pass

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
        v = self.execute(*args, **kwargs)
        assert isinstance(v, dict)
        assert all([parm in v.keys() for parm in self.parameters])
        t = np.array([self.tolerances[parm] for parm in self.parameters])
        w = np.array([self.weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([v[parm] for parm in self.parameters])
        return INDEX_FUNCTIONS[self.functiontype](x, w, t, self.amplitude)  

    
class UtilityFunction(ABC): 
    def __init_subclass__(cls, functionname, functiontype, *args, parameters=[], coefficents=[], **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(parameters, (tuple, list)) and isinstance(coefficents, (tuple, list))
        assert functiontype in UTILITY_FUNCTIONS.keys()
        setattr(cls, 'funcitonname', functionname)
        setattr(cls, 'functiontype', functiontype)
        setattr(cls, 'parameters', tuple(sorted(parameters)))
        setattr(cls, 'coefficents', tuple(coefficents))            

    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass
    def execute(self, *args, **kwargs): return {}

    @property
    def key(self): 
        types = (self.name, self.functiontype)
        coefficents = [(key, value) for key, value in self.__coefficents.items()]
        items = [(parm, self.__subsistences[parm], self.__weights[parm]) for parm in self.parameters]   
        functions = [(parm, self.__functions[parm].key if parm in self.__functions.keys() else None) for parm in self.parameters]
        return hash((tuple(types), tuple(coefficents), tuple(items), tuple(functions)))
    
    def __repr__(self): 
        string = '{}(name={}, functiontype={}, coefficents={}, subsistences={}, weights={})' 
        return string.format(self.__class__.__name__, self.name, self.functiontype, self.__coefficents, self.__subsistences, self.__weights)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, subsistences={}, weights={}, functions={}, coefficents={}, **kwargs): 
        assert isinstance(subsistences, dict) and isinstance(weights, dict)
        assert all([hasattr(self, attr) for attr in ('name', 'functiontype',)])
        self.__subsistences = {parm:subsistences.get(parm, 0) for parm in self.parameters}
        self.__weights = {parm:weights.get(parm, 0) for parm in self.parameters}
        self.__functions = {parm:functions[parm] for parm in self.parameters if parm in functions}        
        self.__coefficents = {coefficent:kwargs[coefficent] for coefficent in self.coefficents}         
        
    def __call__(self, *args, **kwargs):
        nestedkwargs = {parm:func(*args, **kwargs) for parm, func in self.__functions.items()}
        x = self.execute(*args, **nestedkwargs, **kwargs)
        x = np.array([x[parm] for parm in self.parameters])                
        c = [self.__coefficents[coefficent] for coefficent in self.coefficents]
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try: u = UTILITY_FUNCTIONS[self.functiontype](np.subtract(x, s), w, *c)
            except Warning: raise NumericalError(np.subtract(x, s))
        return u

    def derivative(self, filtration, *args, **kwargs):
        filtration = _aslist(filtration)
        nestedkwargs = {parm:func(*args, **kwargs) for parm, func in self.__functions.items()}
        i = self.parameters.index(filtration[0])
        x = self.execute(*args, **nestedkwargs, **kwargs)
        x = np.array([x[parm] for parm in self.parameters])                        
        c = [self.__coefficents[coefficent] for coefficent in self.coefficents]
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w)) 
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try: du = UTILITY_DERIVATIVES[self.functiontype](i, np.subtract(x, s), w, *c)
            except Warning: raise NumericalError(np.subtract(x, s))
        dx = self.__functions[filtration[0]].derivative(filtration[1:], *args, **kwargs) if len(filtration) > 1 else 1       
        return du * dx





        
        