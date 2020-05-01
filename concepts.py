# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 2020
@name:   Field Objects
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


def concept(name, fields):
    assert all([isinstance(field, str) for field in _aslist(fields)])     
     
    def __new__(cls, **kwargs): return super().__new__(cls, *[kwargs.get(field, None) for field in cls._fields])
    def __getitem__(self, key): return self._asdict().items()[key]  
    def __hash__(self): return hash((self.__class__.__name__, *[hash((concept, hash(getattr(self, field)),)) for field in self._fields]))
    
    @property
    def fields(self): return self._fields
    def todict(self): return self._asdict()   
    
    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join([field.lower() for field in _aslist(fields)]))    
    attrs = {'todict':todict, 'fields':fields, '__new__':__new__, '__getitem__':__getitem__, '__hash__':__hash__}
    return type(name, (base,), attrs)







