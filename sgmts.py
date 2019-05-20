# -*- coding: utf-8 -*-
"""
Created on Sat Sept 23 2017
@name    Segment Objects
@author: Jack Kriby Cook

"""

from abc import ABC, abstractmethod
from functools import wraps

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['concat_arguments', 'concat_layers', 'value_segment', 'argument_segment', 'keyword_segment', 'layer_segment']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda values: [values] if not isinstance(values, (list, tuple)) else values


def concat_arguments(items, arg_delimiter, kw_delimiter):
    recursion = lambda item: concat_arguments(item, arg_delimiter=arg_delimiter, kw_delimiter=kw_delimiter)    

    if isinstance(items, dict): return arg_delimiter.join([kw_delimiter.join([recursion(key), recursion(value)]) for key, value in items.items()])
    elif isinstance(items, (tuple, list)): return arg_delimiter.join([recursion(item) for item in items])
    else: return str(items)
       

def concat_layers(items, delimiters, excess_delimiter):
    assert isinstance(items, (list, tuple, int, float, str))
    assert isinstance(delimiters, (list, tuple))
    assert isinstance(excess_delimiter, str)

    delimiters = list(delimiters) + [excess_delimiter]
    recursion = lambda item: concat_layers(item, delimiters=delimiters[1:], excess_delimiter=excess_delimiter)     

    if isinstance(items, (list, tuple)): return delimiters[0].join([recursion(item) for item in items])
    else: return str(items)


class Segment(ABC):
    startchar, endchar = '', ''
    
    @abstractmethod
    def strings(self): pass

    @property
    def items(self): return self.__items
    def __init__(self, items): self.__items = items
    def __str__(self): return ''.join([self.startchar, self.strings, self.endchar]) if self.strings is not None else ''


class Value_Segment(Segment):
    def __init__(self, items=None): 
        assert not isinstance(items, (list, tuple))
        super().__init__(items)
        
    @property
    def strings(self): return str(self.items) if self.items is not None else None


class Argument_Segment(Segment):
    delimiter = ', '
    
    def __init__(self, *items): 
        assert all([isinstance(item, (str, int, float, tuple, list)) for item in items])
        super().__init__(items)        

    @property
    def strings(self): return self.delimiter.join([str(item) for item in self.items]) if self.items else None

    def __add__(self, other): 
        if not isinstance(other, self.__class__): raise TypeError(type(other))
        return self.__class__(*self.items, *other.items)
    

class Keyword_Segment(Segment):   
    arg_delimiter, kw_delimiter = ', ', '='
    
    def __init__(self, **items): 
        assert all([isinstance(item, (str, int, float, tuple, list)) for item in items.values()])
        super().__init__(items)            

    @property
    def strings(self): return self.arg_delimiter.join([self.arg_delimiter.join([self.kw_delimiter.join([key, str(value)]) for value in _aslist(values)]) for key, values in self.items.items()]) if self.items else None

    def __add__(self, other):
        if not isinstance(other, self.__class__): raise TypeError(type(other))
        keys = list(self.items.keys()) + [key for key in other.items.keys() if key not in self.items.keys()]
        return self.__class__(**{key:[*_aslist(self.items.get(key, [])), *_aslist(other.items.get(key, []))] for key in keys})


class Layer_Segment(Segment):
    delimiters, excess_delimiter = [',', '='], ''
    
    def __init__(self, items=[]): 
        assert isinstance(items, (list, tuple))
        super().__init__(items)
        
    @property
    def strings(self): return concat_layers(self.items, delimiters=self.delimiters, excess_delimiter=self.excess_delimiter) 


def segment(base):
    def decorator(function):
        @wraps(function)
        def wrapper(name, *args, startchar='', endchar='', **kwargs):
            assert isinstance(name, str)
            assert isinstance(startchar, str)
            assert isinstance(endchar, str)
            return type(name, (base,), dict(startchar=startchar, endchar=endchar, **function(*args, **kwargs)))
        return wrapper
    return decorator


@segment(Value_Segment)
def value_segment(*args, **kwargs):    
    return dict()
    

@segment(Argument_Segment)
def argument_segment(delimiter, *args, **kwargs):
    assert isinstance(delimiter, str)
    return dict(delimiter=delimiter)
    

@segment(Keyword_Segment)
def keyword_segment(arg_delimiter, kw_delimiter, *args, **kwargs):
    assert isinstance(arg_delimiter, str)
    assert isinstance(kw_delimiter, str)    
    return dict(arg_delimiter=arg_delimiter, kw_delimiter=kw_delimiter)


@segment(Layer_Segment)
def layer_segment(delimiters, excess_delimiter, *args, **kwargs):
    assert isinstance(delimiters, (list, tuple))
    assert isinstance(excess_delimiter, str)
    return dict(delimiters=delimiters, excess_delimiter=excess_delimiter)







        
        
        
        
        
        
        
        
        
        
        
