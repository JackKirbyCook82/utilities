# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 2020
@name:   Field Objects
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['concept', 'layer']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


_DELIMITER = '|'
_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)


def concept(name, fields):
    assert all([isinstance(field, str) for field in _aslist(fields)]) 
    
    def __new__(cls, **kwargs): return super().__new__(cls, **{key:kwargs.get(key, None) for key in cls._fields})
    def __getitem__(self, key): return self._asdict().items()[key]  
    def __hash__(self): return hash((self.__class__.__name__, *[(field, getattr(self, field),) for field in self._fields]))
    def __str__(self): return _DELIMITER.join(['{}={}'.format(field, getattr(self, field)) for field in self._fields])
    def todict(self): return self._asdict()
    
    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join([field.lower() for field in _aslist(fields)]))    
    attrs = {'todict':todict, '__new__':__new__, '__getitem__':__getitem__, '__str__':__str__, '__hash__':__hash__}
    return type(name, (base,), attrs)


def layer(name, concepts):
    assert all([isinstance(field, str) for field in _aslist(concepts)])     
     
    def __new__(cls, **kwargs):
        assert all([isinstance(kwargs.get(x, {}), dict) for x in cls._concepts])      
        getfields = lambda x: [key.lower() for key in kwargs.get(x, {}).keys()]
        getitems = lambda x: {key.lower():value for key, value in kwargs.get(x, {}).items()}
        Concepts = {x:concept(x, fields=getfields(x)) for x in cls._concepts}
        concepts = {x:X(**getitems(x)) for x, X in Concepts.items()}
        return super().__new__(cls, **concepts)
    
    def __getitem__(self, key): return self._asdict().items()[key]  
    def __hash__(self): return hash((self.__class__.__name__, *[hash((concept, hash(getattr(self, concept)),)) for concept in self._fields]))
    def __str__(self): return '\n'.join(['{}[{}]'.format(concept, str(getattr(self, concept))) for concept in self._fields])
    def todict(self): return self._asdict()   
    
    name = uppercase(name)
    base = ntuple(uppercase(name), ' '.join([field.lower() for field in concepts]))    
    attrs = {'todict':todict, '__new__':__new__, '__getitem__':__getitem__, '__str__':__str__, '__hash__':__hash__}
    return type(name, (base,), attrs)




























