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
    'inverted': lambda t, w, x: np.sum(np.divide(np.divide(w, t), x)),
    'tangent': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan(x * np.pi/2))),
    'rtangent': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.tan((1 - x) * np.pi/2))),
    'logarithm': lambda t, w, x: np.sum(np.multiply(np.divide(w, t), np.log(x + 1)))}
UTILITYFUNCTIONS = {
    'cobbdouglas': lambda a, b, c, w, x: a * np.power(np.prod(np.power(np.subtract(x, b), w)), c)}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_normalize = lambda items: np.array(items) / np.sum(np.array(items))


class UtilityIndex(ABC): 
    @abstractmethod
    def execute(self, *args, **kwargs): pass
        
    def __repr__(self): return '{}(amplitude={}, tolerances={})'.format(self.__class__.__name__, self.amplitude, self.tolerances)
    def __hash__(self): return hash((self.__class__.__name__, self.functiontype, self.amplitude, tuple(self.tolerances), tuple(self.parameters), tuple(self.weights),))
    def __len__(self): return len(self.parameters)
   
    def __init__(self, amplitude=1, tolerances={}): 
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
    def register(cls, functiontype, parameter_weights):
        if cls != UtilityIndex: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'create'))
        assert functiontype in INDEXFUNCTIONS.keys()
        assert isinstance(parameter_weights, dict)
        parameters = list(parameter_weights.keys())
        weights = _normalize(list(parameter_weights.values()))
        attrs = dict(parameters=parameters, weights=weights, functiontype=functiontype, function=INDEXFUNCTIONS[functiontype])
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper
    
    @classmethod
    def subclasses(cls): return {subclass.functiontype:subclass for subclass in cls.__subclasses__()}     
    @classmethod
    def create(cls, functiontype, *args, **kwargs): return cls.subclasses()[functiontype](*args, **kwargs)


class UtilityFunction(ABC):
    @abstractmethod
    def execute(self): pass
    
    def __repr__(self): return '{}()'.format(self.__class__.__name__)
    def __hash__(self): return hash((self.__class__.__name__, self.amplitude, self.diminishrate, tuple(self.subsistences), tuple(self.weights), tuple(self.parameters), tuple([hash(index) for index in self.indexes]),))
    
    def __init__(self, parameters, *args, **kwargs):
        assert isinstance(parameters, dict)
        self.parameters, self.indexes = parameters.keys(), parameters.values()
        
    def __len__(self): return len(self.parameters)
    def __call__(self, *args, **kwargs): 
        x = np.array([index(*args, **kwargs) for parameter, index in zip(self.parameters, self.indexes)])
        y = self.exeucte(x, *args, **kwargs)
        return y   

    @classmethod
    def register(cls, functiontype):
        if cls != UtilityFunction: raise NotImplementedError('{}.{}()'.format(cls.__name__, 'create'))
        assert functiontype in UTILITYFUNCTIONS.keys()
        attrs = dict(functiontype=functiontype, function=UTILITYFUNCTIONS[functiontype])
        def wrapper(subclass): return type(subclass.__name__, (subclass, cls), attrs)
        return wrapper   
    
    @classmethod
    def subclasses(cls): return {subclass.functiontype:subclass for subclass in cls.__subclasses__()}     
    @classmethod
    def create(cls, functiontype, *args, **kwargs): return cls.subclasses()[functiontype](*args, **kwargs)


@UtilityFunction.register('cobbdouglas')
class CobbDouglas_UtilityFunction(UtilityFunction):
    @property
    def coefficients(self): return self.amplitude, self.subsistences, self.diminishrate, self.weights    
    
    def __repr__(self): return '{}(amplitude={}, subsistences={}, weights={}, diminishrate={})'.format(self.__class__.__name__, self.amplitude, self.subsistences, self.weights, self.tolerances)
    def __hash__(self): return hash((self.__class__.__name__, tuple(self.parameters), tuple([hash(index) for index in self.indexes]),))
    
    def __init__(self, *args, amplitude=1, subsistences={}, weights={}, diminishrate=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert all([isinstance(items, dict) for items in (subsistences, weights)])
        self.amplitude, self.diminishrate = amplitude, diminishrate
        self.subsistences = np.array([subsistences.get(parm, 0) for parm in self.parameters])
        self.weights = _normalize(np.array([weights.get(parm, 0) for parm in self.parameters]))
        
    def execute(self, x, *args, **kwargs):
        return self.function(*self.coefficients, x)
        
        

    


    






        
        