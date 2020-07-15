# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 2020
@name:   Utility Objects
@author: Jack Kirby Cook

"""

import numpy as np
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
    'ces': lambda x, w, a, d, p, *args: (np.sum(np.multiply(np.power(x, p), w)) **d) * a,
    'linear': lambda x, w, a, *args: np.sum(np.multiply(x, w)) ** a}
UTILITY_DERIVATIVES = {
    'cobbdouglas': lambda i, x, w, a, d, *args: a * w[i] * d * (1/x[i]) * (np.prod(np.power(x, w)) ** d),
    'ces': lambda i, x, w, a, d, p, *args: a * w[i] * d * (x[i]**(p-1)) * (np.sum(np.multiply(x, w) ** p) ** ((d/p)-1)),
    'linear': lambda i, x, w, a, d, *args: a * w[i]}
INDEX_FUNCTIONS = {
    'additive': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), x)) * a,
    'inverted': lambda x, w, t, a: np.sum(np.divide(np.divide(w, t), x)) * a,
    'tangent': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))) * a,
    'rtangent': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))) * a,
    'logarithm': lambda x, w, t, a: np.sum(np.multiply(np.divide(w, t), np.log(x + 1))) * a}


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
        return INDEX_FUNCTIONS[self.functiontype](x, w, t, self.amplitude)  

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
        types = (self.name, self.positivetype, self.negativetype)
        coefficents = [(key, value) for key, value in self.__coefficents.items()]
        items = [(parm, self.__subsistences[parm], self.__weights[parm]) for parm in self.parameters]   
        functions = [(parm, self.__functions[parm].key if parm in self.__functions.keys() else None) for parm in self.parameters]
        return hash((tuple(types), tuple(coefficents), tuple(items), tuple(functions)))
    
    def __repr__(self): 
        string = '{}(name={}, positivetype={}, negativetype={}, coefficents={}, subsistences={}, weights={})' 
        return string.format(self.__class__.__name__, self.name, self.positivetype, self.negativetype, self.__coefficents, self.__subsistences, self.__weights)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, subsistences={}, weights={}, functions={}, coefficents={}, **kwargs): 
        assert isinstance(subsistences, dict) and isinstance(weights, dict)
        assert all([hasattr(self, attr) for attr in ('name', 'positivetype', 'negativetype',)])
        self.__subsistences = {parm:subsistences.get(parm, 0) for parm in self.parameters}
        self.__weights = {parm:weights.get(parm, 0) for parm in self.parameters}
        self.__functions = {parm:functions[parm] for parm in self.parameters if parm in functions}        
        self.__coefficents = {coefficent:kwargs[coefficent] for coefficent in self.coefficents}         
        
    def __call__(self, *args, **kwargs):
        values = self.execute(*args, **kwargs)
        c = [self.__coefficents[coefficent] for coefficent in self.coefficents]
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([self.__functions[parm](*args, **kwargs) if parm in self.__functions.keys() else values[parm] for parm in self.parameters])
        y = np.subtract(x, s)
        if np.all(y > 0): u = UTILITY_FUNCTIONS[self.positivetype](y, w, *c)
        else: u = UTILITY_FUNCTIONS[self.negativetype](np.minimum(y, 0), w, *c)
        return u

    def derivative(self, filtration, *args, **kwargs):
        filtration = _aslist(filtration)
        values = self.execute(*args, **kwargs)
        i = self.parameters.index(filtration[0])
        c = [self.__coefficents[coefficent] for coefficent in self.coefficents]
        s = np.array([self.__subsistences[parm] for parm in self.parameters])
        w = np.array([self.__weights[parm] for parm in self.parameters])
        w = _normalize(w) if sum(w) > 0 else np.ones(w.shape) * (1/len(w))
        x = np.array([self.__functions[parm](*args, **kwargs) if parm in self.__functions.keys() else values[parm] for parm in self.parameters])
        y = np.subtract(x, s)
        if np.all(y > 0): du = UTILITY_DERIVATIVES[self.positivetype](i, y, w, *c)
        else: du = UTILITY_DERIVATIVES[self.negativetype](i, y, w, *c)
        dy = self.__functions[filtration[0]].derivative(filtration[1:], *args, **kwargs) if len(filtration) > 1 else 1
        return du * dy

    @classmethod
    def register(cls, name, positivetype, negativetype, *args, parameters, coefficents, **kwargs):
        if cls != UtilityFunction: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'register'))      
        assert isinstance(parameters, (tuple, list)) and isinstance(coefficents, (tuple, list))
        assert positivetype in UTILITY_FUNCTIONS.keys() and negativetype in UTILITY_FUNCTIONS.keys()       
        attrs = dict(name=name, positivetype=positivetype, negativetype=negativetype, parameters=tuple(sorted(parameters)), coefficents=tuple(sorted(coefficents)))
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper
    








        
        