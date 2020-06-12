# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 2020
@name:   Utility Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod
from collections import OrderedDict as ODict
import numpy as np

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['UtilityIndex', 'UtilityFunction']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.sum(np.array(items))


UTILITYFUNCTIONS = {
    'cobbdouglas': lambda a, d, s, w, x: (np.prod(np.power(np.subtract(x, s), w)) ** d) * a,
    'ces': lambda a, d, s, w, x, p: (np.sum(np.multiply((np.subtract(x, s) ** p), w)) ** (d/p)) * a}
INDEXFUNCTIONS = {
    'inverted': lambda a, t, w, x: np.sum(np.divide(np.divide(w, t), x)),
    'tangent': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))),
    'rtangent': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))),
    'logarithm': lambda a, t, w, x: np.sum(np.multiply(np.divide(w, t), np.log(x + 1)))}


class UtilityIndex(ABC): 
    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass
    @classmethod    
    @abstractmethod
    def function(self, *args, **kwargs): pass

    @property
    def key(self): return hash((self.functionname, self.functiontype, self.amplitude, *[item for item in self.items()],))
    def items(self): return [(parameter, self.tolerances[parameter], self.weight[parameter]) for parameter in sorted(self.parameters)]  
    
    def __repr__(self): 
        string = '{}(functionname={}, functiontype={}, amplitude={}, tolerances={}, weights={})' 
        return string.format(self.__class__.__name__, self.functionname, self.functiontype, self.amplitude, self.tolerances, self.weights)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, tolerances={}, weights={}, **kwargs): 
        assert isinstance(tolerances, dict) and isinstance(weights, dict)
        self.amplitude = amplitude
        self.tolerances = {tolerances.get(parm, 1) for parm in self.parameters}
        self.weights = {weights.get(parm, 1) for parm in self.parameters}
 
    def __call__(self, *args, **kwargs): 
        values = self.execute(*args, **kwargs)
        assert isinstance(values, dict)
        assert all([parameter in values.keys() for parameter in self.parameters])
        t = np.array([self.tolerances[parameter] for parameter in sorted(self.parameters)])
        w = _normalize(np.array([self.weights[parameter] for parameter in sorted(self.parameters)]))
        x = np.array([values[parameter] for parameter in sorted(self.parameters)])
        return self.function(self.amplitude, t, w, x)  
    
    __subclasses = {}          
    @classmethod
    def subclasses(cls): return cls.__subclasses
    @classmethod
    def getsubclass(cls, functionname): return cls.__subclasses[functionname.lower()]   
    def __class_getitem__(cls, functionname): return cls.getsubclass(functionname)
    
    @classmethod
    def register(cls, functionname, functiontype, *args, parameters, **kargs):
        if cls != UtilityIndex: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'register'))      
        assert isinstance(parameters, (tuple, list))
        assert functiontype in INDEXFUNCTIONS.keys()
        attrs = dict(functionname=functionname, functiontype=functiontype, function=INDEXFUNCTIONS[functiontype], parameters=parameters)
        def wrapper(subclass): 
            newsubclass = type(subclass.__name__, (subclass, cls), attrs)
            cls.__subclasses[functionname] = newsubclass
            return newsubclass
        return wrapper
  
    
class UtilityFunction(ABC): 
    @classmethod    
    @abstractmethod
    def create(self, *args, **kwargs): pass
    @classmethod    
    @abstractmethod
    def function(self, *args, **kwargs): pass

    @property
    def key(self): return hash((self.functionname, self.functiontype, self.amplitude, *[(key, value) for key, value in self.coefficents.items()], *[item for item in self.items()],))
    def items(self): return [(parameter, self.subsistences[parameter], self.weight[parameter]) for parameter in sorted(self.parameters)]   
    
    def __repr__(self): 
        content = dict(functionname=self.functionname, functiontype=self.functiontype, amplitude=self.amplitude, subsistences=self.subsistences, weights=self.weights)
        content.update(self.coefficents)
        return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([key, value]) for key, value in content.items()])) 
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, diminishrate=1, subsistences={}, weights={}, indexes={}, **kwargs): 
        assert isinstance(subsistences, dict) and isinstance(weights, dict)
        self.__amplitude, self.__diminishrate = amplitude, diminishrate
        self.__coefficents = ODict([(coefficent, kwargs[coefficent]) for coefficent in self.coefficents])
        self.__subsistences = {subsistences.get(parm, 0) for parm in self.parameters}
        self.__weights = {weights.get(parm, 1) for parm in self.parameters}
        self.__indexes = {indexes[parm] for parm in self.parameters}

    def __call__(self, *args, **kwargs): 
        c = list(self.__coefficents.values())
        s = np.array([self.__subsistences[parameter] for parameter in sorted(self.parameters)])
        w = _normalize(np.array([self.__weights[parameter] for parameter in sorted(self.parameters)]))
        x = np.array([self.__indexes[parameter](*args, **kwargs) for parameter in sorted(self.parameters)])
        return self.function(self.__amplitude, self.__diminishrate, s, w, x, *c)

    __subclasses = {}          
    @classmethod
    def subclasses(cls): return cls.__subclasses
    @classmethod
    def getsubclass(cls, functionname): return cls.__subclasses[functionname.lower()]   
    def __class_getitem__(cls, functionname): return cls.getsubclass(functionname)
    
    @classmethod
    def register(cls, functionname, functiontype, *args, parameters, coefficents, **kwargs):
        if cls != UtilityFunction: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'register'))      
        assert isinstance(parameters, (tuple, list)) and isinstance(coefficents, (tuple, list))
        assert functiontype in UTILITYFUNCTIONS.keys()       
        attrs = dict(functionname=functionname, functiontype=functiontype, function=UTILITYFUNCTIONS[functiontype], parameters=parameters, coefficents=coefficents)
        def wrapper(subclass): 
            newsubclass = type(subclass.__name__, (subclass, cls), attrs)
            cls.__subclasses[functionname] = newsubclass
            return newsubclass
        return wrapper
    








        
        