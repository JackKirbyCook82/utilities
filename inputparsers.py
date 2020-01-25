# -*- coding: utf-8 -*-
"""
Created on Mon Sept 23 2019
@name:   Input Controllers
@author: Jack Kirby cook

"""

import json

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['InputParser']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


class InputParser(object):
    def __init__(self, assignproxy='=', spaceproxy='_', parsers={}):
        self.__spaceproxy = spaceproxy
        self.__assignproxy = assignproxy
        self.__parsers = parsers
        self.reset()
    
    @property
    def inputArgs(self): return self.__inputArgs
    @property
    def inputParms(self): return self.__inputParms    
    
    def reset(self):
        self.__inputArgs = []
        self.__inputParms = {}

    def delete(self, *items):
        for item in items:
            if isinstance(item, int): self.__inputArgs.pop(item)
            elif isinstance(item, str): self.__inputParms.pop(item)
            else: raise TypeError(type(item))

    def __setitem__(self, item, value):
        if isinstance(item, int): self.__inputArgs.insert(item, value)
        elif isinstance(item, str): self.__inputParms[item] = value
        else: raise TypeError(type(item)) 
        
    def __getitem__(self, item):
        if isinstance(item, int): return self.__inputArgs[item] if -len(self.__inputArgs) <= item < len(self.__inputArgs) else None
        elif isinstance(item, str): return self.__inputParms.get(item, None)
        else: raise TypeError(type(item))

    def __repr__(self): return "{}(assignproxy='{}', spaceproxy='{}')".format(self.__class__.__name__, self.__assignproxy, self.__spaceproxy)
    def __str__(self):   
        argsjsonstr = json.dumps(self.inputArgs, sort_keys=False, indent=3, separators=(',', ' : '), default=str)     
        parmsjsonstr = json.dumps(self.inputParms, sort_keys=False, indent=3, separators=(',', ' : '), default=str)            
        return '\n'.join([' '.join(['Input Arguments', argsjsonstr]), ' '.join(['Input Parameters', parmsjsonstr])]) 
            
    def __call__(self, *sysArgs):
        inputArgs, inputParms = [], {}
        for sysArg in sysArgs: 
            if self.__assignproxy in sysArg:
                try: key, value = sysArg.split(self.__assignproxy)[0], sysArg.split(self.__assignproxy)[1]
                except IndexError: key, value = sysArg.split(self.__assignproxy)[0], None
                inputParms[key.replace(self.__spaceproxy, " ")] = value.replace(self.__spaceproxy, " ") 
            else: inputArgs.append(sysArg) 
        for key, parser in self.__parsers.items():
            if key in inputParms.keys(): 
                inputParms[key] = parser(inputParms[key])
        self.__inputArgs.extend(inputArgs)
        self.__inputParms.update(inputParms)

    def addparser(self, key, parser):
        self.__parsers[key] = parser
















    