# -*- coding: utf-8 -*-
"""
Created on Sat May 4 2019
@name:    Quantity Objects
@author: Jack Kirby Cook

"""

import numpy as np

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Multiplier', 'Unit', 'Heading']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


MULTIPLIERS = {'M':1000000, 'K':1000, '%':0.01, '1':1, '':1}

_aslist = lambda items: ([items] if not isinstance(items, (list, tuple)) else items)
_asstr = lambda items: (str(item) for item in items)
_filter = lambda items: (item for item in items if item is not None)


def sametype(function):
    def wrapper(self, other):
        if type(self) != type(other): raise TypeError(' != '.join([str(type(self).__name__), str(type(other).__name__)]))
        return function(self, other)
    return wrapper   


class Quantity(object):
    divchar = '/'
    multchar = '·'
    defaultchar = '1'
    quantityformat = '{}'
    
    def __init__(self, ofquantity=[], byquantity=[]): self.__ofquantity, self.__byquantity = self.__prep(ofquantity, byquantity)     
    def __str__(self): return self.quantityformat.format(self.string) if self else ''

    @property
    def string(self): 
        if not any([self.ofquantity, self.byquantity]): return ''
        elif not self.byquantity: return self.numerator
        else: return self.divchar.join([self.numerator, self.denominator])
    
    @property
    def numerator(self): return self.multchar.join(self.ofquantity) if self.ofquantity else self.defaultchar
    @property
    def denominator(self): return self.multchar.join(self.byquantity) if self.byquantity else self.defaultchar
    
    @staticmethod
    def __prep(ofquantity, byquantity):
        generator = lambda items: _asstr(_filter(_aslist(items)))
        ofquantity, byquantity = [item for item in generator(ofquantity)], [item for item in generator(byquantity)]
        return tuple([item for item in ofquantity if item not in byquantity]), tuple([item for item in byquantity if item not in ofquantity])

    @property
    def ofquantity(self): return self.__ofquantity
    @property
    def byquantity(self): return self.__byquantity

    def __eq__(self, other): 
        if not isinstance(other, type(self)): return False
        return all([self.ofquantity == other.ofquantity, self.byquantity == other.byquantity])
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash((self.__ofquantity, self.__byquantity,))
    def __bool__(self): return set(self.numerator) != set(self.denominator)
    
    @sametype
    def __mul__(self, other): return self.__class__([*self.ofquantity, *other.ofquantity], [*self.byquantity, *other.byquantity])
    @sametype
    def __truediv__(self, other): return self.__class__([*self.ofquantity, *other.byquantity], [*self.byquantity, *other.ofquantity])

    @classmethod
    def fromstr(cls, string): 
        function = lambda items: [item for item in items.split(cls.multchar) if item]
        try: ofquantity, byquantity = string.split(cls.divchar)
        except ValueError: ofquantity, byquantity = string.split(cls.divchar)[0], ''
        return cls(function(ofquantity), function(byquantity))

    @classmethod
    def register(cls, quantityformat):  
        def wrapper(subclass):
            name = subclass.__name__
            bases = (subclass, cls)
            newsubclass = type(name, bases, dict(quantityformat=quantityformat))
            return newsubclass
        return wrapper  
        

@Quantity.register(quantityformat=' {}')
class Unit: pass

 
@Quantity.register(quantityformat='{}')
class Heading: pass


@Quantity.register(quantityformat='{}')
class Multiplier:
    @property
    def num(self):
        values = lambda x: [1, *[MULTIPLIERS[item] for item in x]]
        return np.prod(values(self.ofquantity)) / np.prod(values(self.byquantity))

    @sametype
    def __lt__(self, other): return self.num < other.num
    @sametype
    def __gt__(self, other): return self.num > other.num
        
    @sametype
    def __le__(self, other): return self.num <= other.num
    @sametype
    def __ge__(self, other): return self.num >= other.num











