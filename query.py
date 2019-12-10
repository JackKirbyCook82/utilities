# -*- coding: utf-8 -*-
"""
Created on Weds Sept 11 2019
@name:   Query Object
@author: Jack Kirby cook

"""

import json
import pandas as pd

from utilities.dataframes import dataframe_fromfile, dataframe_parser

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['QuerySgmts', 'Query', 'EmptyQueryError']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""
    

_aslist = lambda items: [item for item in items] if hasattr(items, '__iter__') and not isinstance(items, str) else [items]
_isnull = lambda value: pd.isnull(value) if not isinstance(value, (list, tuple, dict)) else False
_filterempty = lambda items: [item for item in _aslist(items) if item]


class QuerySgmts(object):
    def __init__(self, universe=None, index=None, header=None, name=None, **scope):
        self.__name = name
        assert isinstance(scope, dict)
        self.__universe = universe
        self.__index = index
        self.__header = header
        self.__scope = {key:_aslist(value) for key, value in scope.items()}
        
    def todict(self): return dict(universe=self.universe, index=self.index, header=self.header, scope=self.scope)   
    def reset(self): self.__universe, self.__index, self.__header, self.__scope = None, None, None, {}
    def update(self, universe=None, index=None, header=None, **scope): 
        self.__universe = universe if universe else self.__universe
        self.__index = index if index else self.__index
        self.__header = header if header else self.__header
        self.__scope.update(scope)
   
    def __str__(self): 
        namestr = '{name} Query'.format(name=self.name) if self.name else 'Query'
        contentstr = {key:self[key] for key in ('universe', 'header', 'index') if self[key]}
        scopestr = {key:value for key, value in self.scope.items() if value}
        jsonstr = json.dumps({**contentstr, **scopestr}, sort_keys=False, indent=3, separators=(',', ' : '))
        return ' '.join([namestr, jsonstr])
       
    def __setitem__(self, key, value): self.update(**{key:value})        
    def __getitem__(self, key):
        if key == 'universe': return self.universe
        elif key == 'index': return self.index
        elif key == 'header': return self.header
        elif key == 'scope': return self.scope
        else: return self.__scope.get(key, None)
           
    @property
    def name(self): return self.__name
    @property
    def universe(self): return self.__universe
    @property
    def index(self): return self.__index
    @property
    def header(self): return self.__header
    @property
    def scope(self): return self.__scope    
    

class EmptyQueryError(Exception): pass


class Query(object):
    @property
    def name(self): return self.__name
    def __repr__(self): return "{}(file='{}', name='{}')".format(self.__class__.__name__, self.__file, self.__name)  
    def __str__(self): return str(self.__querysgmts)
    
    def __init__(self, file, parsers, name=None):
        self.__name = name
        self.__file = file
        self.__parsers = parsers
        data = dataframe_fromfile(file, index=None, header=0, forceframe=True)  
        data = dataframe_parser(data, parsers=parsers, defaultparser=None)
        data.set_index('tableID', drop=True, inplace=True)                          
        self.__data = data     
        self.__querysgmts = QuerySgmts(name=name)

    def setuniverse(self, universe): self.__querysgmts.update(universe=universe)
    def setindex(self, index): self.__querysgmts.update(index=index)
    def setheader(self, header): self.__querysgmts.update(header=header)
    def setscope(self, **scope): self.__querysgmts.update(**scope)  

    def __contains__(self, tableID): return tableID in self.data.index.values
    def __getitem__(self, key): return self.__querysgmts[key]
    def __setitem__(self, key, value): self.__querysgmts[key] = value

    @property
    def tableIDs(self): return list(self.data.index.values)
    def __call__(self, tableID): return self.asdict()[tableID] 
    def asdict(self): return {tableID:{key:(value if not _isnull(value) else None) for key, value in values.items()} for tableID, values in self.data.transpose().to_dict().items()}
    def reset(self): self.__querysgmts.reset()
    def display(self): 
        data = self.data[['universe', 'index', 'header', 'scope']]
        data.loc[:, 'scope'] = data.loc[:, 'scope'].apply(lambda x: ', '.join(['='.join([key, x[key]]) for key in set(x.keys())]))         
        return data

    @property
    def data(self):
        data = self.__data.copy()
        for key in ('universe', 'index', 'header'):
            if self[key]: data = data[data[key]==self[key]]   
        if data.empty: raise EmptyQueryError()   
        for key, value in self['scope'].items(): 
            if value: data = data[data['scope'].apply(lambda x: x[key] == value if key in x.keys() else False)]
        if data.empty: raise EmptyQueryError()
        data = data.dropna(axis=0, how='all')
        data.columns.name = '{} Query'.format(self.name) if self.name else 'Query'
        try: return data.to_frame()
        except: return data   
             
    


        
            
    
