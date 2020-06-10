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
__all__ = ['UtilityIndex', 'CobbDouglas_UtilityFunction']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


INDEXFUNCTIONS = {
    'inverted': lambda t, w, x: np.sum(np.divide(np.divide(w, t), x)),
    'tangent': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))),
    'rtangent': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))),
    'logarithm': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.log(x + 1)))}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.sum(np.array(items))


class UtilityIndex(ABC): 
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def key(self): return (self.functionname, self.functiontype, self.amplitude, *[item for item in self.items()],)
    def items(self): return [(parameter, weight, tolerance) for parameter, weight, tolerance in zip(self.parameters, self.weights, self.tolerances)]  
    
    def __repr__(self): 
        string = '{}(functionname={}, functiontype={}, amplitude={}, tolerances={}, weights={})' 
        tolerancesDict = {parameter:tolerance for parameter, tolerance in zip(self.parameters, self.tolerances)}
        weightsDict = {parameter:weight for parameter, weight in zip(self.parameters, self.weights)}
        return string.format(self.__class__.__name__, self.functionname, self.functiontype, self.amplitude, tolerancesDict, weightsDict)
        
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, tolerances={}, **kwargs): 
        assert isinstance(tolerances, dict)
        self.amplitude = amplitude
        self.tolerances = np.array([tolerances.get(parm, 1) for parm in self.parameters])    
 
    def __call__(self, *args, **kwargs): 
        parameters = self.execute(*args, **kwargs)
        assert isinstance(parameters, dict)
        assert all([parameter in parameters.keys() for parameter in self.parameters])
        x = np.array([parameters[parameter] for parameter in self.parameters])
        y = self.amplitude * self.function(self.tolerances, self.weights, x)
        return y
 
    @classmethod
    def create(cls, functionname, functiontype, parameter_weights):
        if cls != UtilityIndex: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'create'))
        assert functiontype in INDEXFUNCTIONS.keys()
        assert isinstance(parameter_weights, dict)
        parameters = list(parameter_weights.keys())
        weights = _normalize(list(parameter_weights.values()))
        attrs = dict(parameters=parameters, weights=weights, functionname=functionname, functiontype=functiontype, function=INDEXFUNCTIONS[functiontype])
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper
    

class CobbDouglas_UtilityFunction(object):
    @property
    def key(self): return (self.__class__.__name__, self.amplitude, self.diminishrate, *[(*item[:-1], item[-1].key) for item in self.items()],)
    def items(self): return [(parameter, subsistence, weight, index) for parameter, subsistence, weight, index in zip(self.parameters, self.subsistences, self.weights, self.indexes,)]  
    def function(self, a, b, c, w, x): return a * np.power(np.prod(np.power(np.subtract(x, b), w)), c)
    
    def __repr__(self): 
        string = '{}(amplitude={}, diminishrate={}, subsistences={}, weights={}, indexes={})'
        subsistencesDict = {parameter:subsistence for parameter, subsistence in zip(self.parameters, self.subsistences)}
        weightsDict = {parameter:weight for parameter, weight in zip(self.parameters, self.weights)}
        indexesDict = {parameter:repr(index) for parameter, index in zip(self.parameters, self.indexes)}
        return string.format(self.__class__.__name__, self.functionname, self.functiontype, self.amplitude, self.diminishrate, subsistencesDict, weightsDict, indexesDict)
    
    def __len__(self): return len(self.parameters)
    def __init__(self, *args, amplitude=1, diminishrate=1, subsistences={}, weights={}, indexes, **kwargs):
        assert all([isinstance(items, dict) for items in (subsistences, weights, indexes)])
        self.amplitude, self.diminishrate = amplitude, diminishrate
        self.parameters, self.indexes = indexes.keys(), indexes.values()
        self.subsistences = np.array([subsistences.get(parm, 0) for parm in self.parameters])
        self.weights = np.array([weights.get(parm, 0) for parm in self.parameters])        
        if np.all(self.weights == 0): self.weights = np.ones(self.weights.shape)
        else: self.weights = _normalize(self.weights) 

    def __call__(self, *args, **kwargs): 
        x = np.array([index(*args, **kwargs) for parameter, index in zip(self.parameters, self.indexes)])
        y = self.function(self.amplitude, self.subsistences, self.diminishrate, self.weights, x)
        return y  


    






        
        