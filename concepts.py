# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 2020
@name:   Concept Objects
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple
from numbers import Number

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['concept']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_defaultfunction = lambda x, *args, **kwargs: x


def concept(name, fields, function=_defaultfunction, fieldfunctions={}): 
    assert isinstance(fields, list) and isinstance(fieldfunctions, dict)
    assert all([hasattr(fieldfunction, '__call__') for fieldfunction in fieldfunctions.values()])
    assert hasattr(function, '__call__')
    
    fields = [field.lower() for field in [*fields, *fieldfunctions.keys()]]
    functions = {field:fieldfunctions.get(field, function) for field in fields}
       
    def todict(self): return self._asdict()  
    def __hash__(self): return hash(tuple([(field, hash(value),) for field, value in self.todict().items()]))
    def __repr__(self): 
        content = {key:(str(value) if isinstance(value, (str, Number)) else repr(value)) for key, value in self.todict().items()}
        return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([key, value]) for key, value in content.items()]))
    
    @classmethod
    def combine(cls, other):
        assert hasattr(other, '_fields') and hasattr(other, '_functions')
        newfieldfunctions = {fields:cls._functions.get(field, other._functions[field]) for field in set([*cls._fields, *other._fields])}
        return concept(cls.__name__, fieldfunctions=newfieldfunctions)
    
    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join(fields))      
    attrs = {'_functions':functions, '__repr__':__repr__, 'combine':combine,  'todict':todict}
    Concept = type(name, (base,), attrs)

    def __new__(cls, items, *args, **kwargs): 
        assert isinstance(items, dict)
        items = {field:items[field] for field in cls._fields}
        items = {field:function(items[field], *args, **kwargs) for field, function in cls._functions.items()}
        return super(Concept, cls).__new__(cls, **items) 

    def __getitem__(self, item): 
        if isinstance(item, (int, slice)): return super(Concept, self).__getitem__(item)
        elif isinstance(item, str): return getattr(self, item)
        else: raise TypeError(type(item))

    setattr(Concept, '__getitem__', __getitem__)
    setattr(Concept, '__new__', __new__)
    return Concept
    










