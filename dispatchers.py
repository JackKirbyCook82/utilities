# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 2017
@name    Function Dispathers
@author: Jack Kriby Cook

"""

from functools import update_wrapper
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['clskey_singledispatcher']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def clskey_singledispatcher(key): 
    def decorator(mainfunc):
        _registry = ODict()
    
        def wrapper(self, *args, **kwargs): return _registry.get(kwargs[key], mainfunc)(self, *args, **kwargs)
        def update(regfunc, *values): _registry.update({value:regfunc for value in values}) 
        def registry(): return _registry
    
        def register(*values): 
            def register_decorator(regfunc): 
                update(regfunc, *values) 
                def register_wrapper(*args, **kwargs): 
                    return regfunc(*args, **kwargs) 
                return register_wrapper 
            return register_decorator 
            
        wrapper.register = register
        wrapper.registry = registry
        update_wrapper(wrapper, mainfunc)
        return wrapper
    return decorator
    



