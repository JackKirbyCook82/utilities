# -*- coding: utf-8 -*-
"""
Created on Fri Feb 8 2019
@name:   String Functions
@author: Jack Kirby Cook

"""

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['uppercase']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_UPPERCHARS = '/*+-_ |('


def uppercase(string, index=0, withops=False):
    if string is None: return string
    if withops: return string[index].upper() + ''.join([item.upper() if string[indx] in _UPPERCHARS else item for item, indx in zip(string[index+1:], range(len(string[index+1:])))])
    else: return string[index].upper() + string[index+1:]