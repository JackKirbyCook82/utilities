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
    
    fields = [field.lower() for field in [*fields, *fieldfunctions.values()]]
    functions = {field:fieldfunctions.get(field, function) for field in fields}
       
    def __getitem__(self, field): return getattr(self, field)
    def __hash__(self): return hash((self.__class__.__name__, *[(field, hash(value)) for field, value in self.todict().items()],))
    def __repr__(self): 
        content = {field:value for field, value in self.todict().items()}
        content = {key:(value if isinstance(value, (str, Number)) else repr(value)) for key, value in content.items()}
        return '{}({})'.format(self.__class__.__name__, ', '.join(['='.join([key, value]) for key, value in content.items()]))
    
    def __new__(cls, items, *args, **kwargs): 
        assert isinstance(items, dict)
        items = {field:items.get(field, None) for field in cls._fields}
        items = {field:(function(items[field], *args, **kwargs) if items[field] is not None else None) for field, function in cls._functions.items()}
        return super(cls, cls).__new__(cls, **items)      
    
    @classmethod
    def combine(cls, other):
        assert hasattr(other, '_fields') and hasattr(other, '_functions')
        newfieldfunctions = {fields:cls._functions.get(field, other._functions[field]) for field in set([*cls._fields, *other._fields])}
        return concept(cls.__name__, fieldfunctions=newfieldfunctions)
    
    def todict(self): return self._asdict()    
    
    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join(fields))       
    attrs = {'__new__':__new__, '__getitem__':__getitem__, '__repr__':__repr__, '__hash__':__hash__,
            'combine':combine, '_functions':functions, 'todict':todict}
    return type(name, (base,), attrs)












