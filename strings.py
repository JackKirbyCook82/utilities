# -*- coding: utf-8 -*-
"""
Created on Fri Feb 8 2019
@name:   String Functions
@author: Jack Kirby Cook

"""

import numpy as np

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['uppercase', 'dictstring', 'liststring']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_OPERATIONS = r"/*+-_ |(@%"


def uppercase(string, index=0, withops=False):
    if string is None: return string
    if withops: return string[index].upper() + ''.join([item.upper() if string[indx] in _OPERATIONS else item for item, indx in zip(string[index+1:], range(len(string[index+1:])))])
    else: return string[index].upper() + string[index+1:]
    
    
strformating = lambda x: uppercase(x, withops=True)
intformating = lambda x: '{:.0f}'.format(x)
floatformating = lambda x: '{:.2f}'.format(x)


def dictstring(items):
    assert isinstance(items, dict)
    functions = {int:intformating, float:floatformating, str:strformating, np.float64:floatformating, np.int64:intformating}
    return ', '.join([key + '=' + functions[type(value)](value) for key, value in items.items()])
    

def liststring(items):
    assert isinstance(items, list)
    functions = {int:intformating, float:floatformating, str:strformating, np.float64:floatformating, np.int64:intformating}
    return ', '.join([functions[type(item)](item) for item in items.items()])