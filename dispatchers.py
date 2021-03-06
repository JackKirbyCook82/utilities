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
__all__ = ['key_singledispatcher', 'clskey_singledispatcher', 'type_singledispatcher', 'clstype_singledispatcher', 'keyword_singledispatcher']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


def clskey_singledispatcher(mainfunc):
    _registry = ODict()
    
    def update(regfunc, *keys): _registry.update({key:regfunc for key in keys})
    def registry(): return _registry
    
    def register(*keys): 
        def register_decorator(regfunc): 
            update(regfunc, *keys) 
            def register_wrapper(*args, **kwargs): 
                return regfunc(*args, **kwargs) 
            return register_wrapper 
        return register_decorator 

    def wrapper(self, key, *args, **kwargs): 
        func = _registry.get(key, mainfunc)
        if key in _registry.keys(): return func(self, *args, **kwargs)         
        else: return func(self, key, *args, **kwargs)

    wrapper.register = register 
    wrapper.registry = registry
    update_wrapper(wrapper, mainfunc)
    return wrapper


def key_singledispatcher(mainfunc):
    _registry = ODict()
    
    def update(regfunc, *keys): _registry.update({key:regfunc for key in keys})
    def registry(): return _registry
    
    def register(*keys): 
        def register_decorator(regfunc): 
            update(regfunc, *keys) 
            def register_wrapper(*args, **kwargs): 
                return regfunc(*args, **kwargs) 
            return register_wrapper 
        return register_decorator 

    def wrapper(key, *args, **kwargs): 
        func = _registry.get(key, mainfunc)
        if key in _registry.keys(): return func(*args, **kwargs)         
        else: return func(key, *args, **kwargs)

    wrapper.register = register 
    wrapper.registry = registry
    update_wrapper(wrapper, mainfunc)
    return wrapper


def keyword_singledispatcher(keyword):
    def decorator(mainfunc):
        _registry = {}
        
        def update(regfunc, *keys): _registry.update({key:regfunc for key in keys})
        def registry(): return _registry
        
        def register(*keys): 
            def register_decorator(regfunc): 
                update(regfunc, *keys) 
                def register_wrapper(*args, **kwargs): 
                    return regfunc(*args, **kwargs) 
                return register_wrapper 
            return register_decorator 
    
        def wrapper(*args, **kwargs):
            return _registry.get(kwargs[keyword], mainfunc)(*args, **kwargs)       
            
        wrapper.register = register 
        wrapper.registry = registry
        update_wrapper(wrapper, mainfunc)
        return wrapper
    return decorator


def clstype_singledispatcher(mainfunc):
    _registry = ODict()
    
    def update(regfunc, *types): _registry.update({item:regfunc for item in types})
    def registry(): return _registry
    
    def register(*types): 
        def register_decorator(regfunc): 
            update(regfunc, *types) 
            def register_wrapper(*args, **kwargs): 
                return regfunc(*args, **kwargs) 
            return register_wrapper 
        return register_decorator 

    def wrapper(self, arg, *args, **kwargs): 
        func = _registry.get(type(arg), mainfunc)
        return func(self, arg, *args, **kwargs)         

    wrapper.register = register 
    wrapper.registry = registry
    update_wrapper(wrapper, mainfunc)
    return wrapper


def type_singledispatcher(mainfunc):
    _registry = ODict()
    
    def update(regfunc, *types): _registry.update({item:regfunc for item in types})
    def registry(): return _registry
    
    def register(*types): 
        def register_decorator(regfunc): 
            update(regfunc, *types) 
            def register_wrapper(*args, **kwargs): 
                return regfunc(*args, **kwargs) 
            return register_wrapper 
        return register_decorator 

    def wrapper(arg, *args, **kwargs): 
        func = _registry.get(type(arg), mainfunc)
        return func(arg, *args, **kwargs)         

    wrapper.register = register 
    wrapper.registry = registry
    update_wrapper(wrapper, mainfunc)
    return wrapper



