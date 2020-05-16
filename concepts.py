# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 2020
@name:   Concept Objects
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple

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
    
    fields = [field.lower() for field in [*fields, *fieldfunctions.values()]]
    functions = {field:fieldfunctions.get(field, function) for field in fields}
    
    def __new__(cls, **kwargs): 
        kwargs = {kwargs.get(field, None) for field in cls._fields}
        kwargs = {cls._functions[field](value) if value is not None else value for field, value in kwargs.items()}
        return super().__new__(cls, **kwargs)      
    
    @classmethod
    def combine(cls, other):
        assert hasattr(other, '_fields') and hasattr(other, '_functions')
        newfieldfunctions = {fields:cls._functions.get(field, other._functions[field]) for field in set([*cls._fields, *other._fields])}
        return concept(cls.__name__, fieldfunctions=newfieldfunctions)
    
    def __getitem__(self, key): return self.__getattr__(key)
    def __hash__(self): return hash((self.__class__.__name__, *[hash(getattr(self, field)) for field in self._fields],))
    def todict(self): return self._asdict()    

    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join(fields))    
    attrs = {'combine':combine, '_functions':functions, 'todict':todict, '__new__':__new__, '__getitem__':__getitem__, '__hash__':__hash__}
    return type(name, (base,), attrs)












