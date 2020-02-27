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


INDEXFUNCTIONS = {
    'inverted': lambda a, b, w, x: a / np.sum(np.multiply(w, np.divide(x, b))),
    'tangent': lambda a, b, w, x: a * np.multiply(w, np.tan((np.pi/2) * np.divide(x, b))), 
    'logarithm': lambda a, b, w, x: a * np.log(np.sum(np.multiply(w, np.divide(x, b))) + 1)}   
UTILITYFUNCTIONS = {
    'cobbdouglas': lambda a, b, c, w, x: a * np.power(np.prod(np.power(np.subtract(x, b), w)), c)}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.linalg.norm(np.array(items))


class UtilityIndex(ABC): 
    @property
    def coefficients(self): return self.amplitude, self.tolerances
    @abstractmethod
    def execute(self, *args, **kwargs): pass
        
    def __repr__(self): return '{}(amplitude={}, tolerances={})'.format(self.__class__.__name__, self.amplitude, self.tolerances)
    def __init__(self, amplitude=1, tolerances={}): 
        assert isinstance(tolerances, dict)
        self.amplitude = amplitude
        self.tolerances = np.array([tolerances.get(parm, 1) for parm in self.parameters])    
 
    def __len__(self): return len(self.parameters)
    def __call__(self, *args, **kwargs): 
        parameters = self.execute(*args, **kwargs)
        assert all([parameter in parameters.keys() for parameter in self.parameters])
        x = np.array([parameters[parameter] for parameter in self.parameters])
        y = self.function(*self.coefficients, self.weights, x)
        return y
       
    @classmethod
    def create(cls, functiontype, parameter_weights):
        if cls != UtilityIndex: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'create'))
        assert isinstance(parameter_weights, dict)
        parameters = list(parameter_weights.keys())
        weights = _normalize(parameter_weights.values())
        attrs = dict(parameters=parameters, weights=weights, functiontype=functiontype, function=INDEXFUNCTIONS[functiontype])
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper


class UtilityFunction(ABC):
    @abstractmethod
    def coefficients(self, *args, **kwargs): pass    

    def __len__(self): return len(self.parameters)
    def __call__(self, *args, **kwargs): 
        x = np.array([index(*args, **kwargs) for parameter, index in zip(self.parameters, self.indexes)])
        y = self.function(*self.coefficients, self.weights, x)
        return y    

    @classmethod
    def create(cls, functiontype):
        if cls != UtilityFunction: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'create'))   
        attrs = dict(functiontype=functiontype, function=UTILITYFUNCTIONS[functiontype])
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper



@UtilityFunction.create('cobbdouglas')
class CobbDouglas_UtilityFunction(UtilityFunction):
    @property
    def coefficients(self): return self.amplitude, self.subsistences, self.diminishrate    
    
    def __repr__(self): return '{}(amplitude={}, subsistences={}, weights={}, diminishrate={})'.format(self.__class__.__name__, self.amplitude, self.subsistences, self.weights, self.tolerances)
    def __init__(self, parameters, amplitude=1, subsistences={}, weights={}, diminishrate=1):
        assert all([isinstance(items, dict) for items in (parameters, subsistences, weights)])
        self.parameters, self.indexes = parameters.keys(), parameters.values()
        self.amplitude, self.diminishrate = amplitude, diminishrate
        self.subsistences = np.array([subsistences.get(parm, 0) for parm in self.parameters])
        self.weights = _normalize(np.array([weights.get(parm, 0) for parm in self.parameters]))
        
    



    






        
        