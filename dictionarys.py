# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Dictionary Objects
@author: Jack Kriby Cook

"""

from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['SliceOrderedDict']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


class SliceOrderedDict(ODict):   
    def __hash__(self): return hash(tuple([(key, value) for key, value in self.items()]))
    
    def __getitem__(self, key): 
        if isinstance(key, str): return super().__getitem__(key)        
        elif isinstance(key, slice): return self.__readslice(key)
        elif isinstance(key, int): return self.__retrieve(key, False)
        else: raise TypeError(type(key))

    def __readslice(self, key):
        start, stop = key.start, key.stop 
        if start is None: start = 0 
        if stop is None: stop = len(self) 
        if stop < 0: stop = len(self) + stop 

        newinstance = self.__class__()
        for index, key in enumerate(self.keys()): 
            if start <= index < stop: 
                newinstance[key] = self[key] 
        return newinstance 

    def __readint(self, index):
        if index >= 0: return self.__readslice( slice( index, index + 1 ) )
        else: return self.__readslice( slice( len(self) + index, len(self) + index + 1 ) )

    def __retrieve(self, key, pop):
        if abs(key + 1 if key < 0 else key) >= len(self): raise IndexError(key)
        key, value = list(*self.__readint(key).items())
        if pop: del self[key]
        return value

    def pop(self, key, default=None):
        if isinstance(key, str): return super().pop(key, default)
        elif isinstance(key, int): return self.__retrieve(key, pop=True)
        else: raise TypeError(type(key))

    def get(self, key, default=None):
        if isinstance(key, str): return super().get(key, default)
        elif isinstance(key, int): return self.__retrieve(key, pop=False)
        else: raise TypeError(type(key))
        
    def update(self, others, inplace=True):
        if not isinstance(others, dict): raise TypeError(type(others))
        updated = [(key, others.pop(key, value)) for key, value in self.items()]
        added = [(key, value) for key, value in others.items()]
        if not inplace: return self.__class__(updated + added)
        self = self.__class__(updated + added)
        return self







