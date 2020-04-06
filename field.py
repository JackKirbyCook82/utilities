# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 2020
@name:   Field Objects
@author: Jack Kirby Cook

"""

from collections import namedtuple as ntuple
import json

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['field']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


def field(name, attributes):
    def __new__(cls, **kwargs):
        kwargs = {key.lower():value for key, value in kwargs.items()}
        layer = lambda key, layerkwargs: field(key, *layerkwargs.keys())(**layerkwargs)
        content = {key:layer(key, value) if isinstance(value, dict) else value for key, value in kwargs.items() if key in cls._fields}
        content = {key:content.get(key, None) for key in cls._fields}
        super().__new__(cls, **content)
    
    def todict(self): return {key:value.todict() if hasattr(value, '_fields') else value for key, value in self._asdict().items()} 
    def jsonstr(self): return json.dumps(self.todict(), sort_keys=True, indent=3, separators=(',', ' : '), default=str) 
    def __getitem__(self, key): return self._asdict().items()[key]      
    def __str__(self): return self.jsonstr()
    def __hash__(self): return hash((self.__class__.__name__, self.jsonstr,))
     
    base = ntuple(uppercase(name), ' '.join([attribute.lower() for attribute in attributes]))
    attrs = {'todict':todict, 'jsonstr':jsonstr, '__new__':__new__, '__getitem__':__getitem__, '__str__':__str__, '__hash__':__hash__}
    return type(name, (base,), attrs)































