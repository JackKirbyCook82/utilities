# -*- coding: utf-8 -*-
"""
Created on Weds Sept 4 2019
@name:   Tree/Node/Renderer Objects
@author: Jack Kirby Cook

"""

import json

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Node', 'Tree', 'Renderer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_STYLES = {'double':dict(branch="╠══", terminate="╚══", run="║  ", blank="   "),
           'single':dict(branch="├──", terminate="└──", run="│  ", blank="   "),
           'single_round':dict(branch="├──", terminate="╰──", run="│  ", blank="   ")}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Node(object):
    def __init__(self, key, children=[], parent=None):
        self.__key = key
        self.__children = []
        if _aslist(children): 
            assert all([isinstance(child, type(self)) for child in _aslist(children)])    
            self.__children = self.__children + _aslist(children)
        if parent is not None: 
            assert isinstance(parent, type(self))
            parent.addchild(self)
                    
    @property
    def key(self): return self.__key                
    @property
    def children(self): return self.__children    
    @property
    def offspring(self): 
        generator = iter(self)
        next(generator)
        return [node for node in generator]
       
    def __len__(self): return len(self.offspring)      
    def __iter__(self):
        yield self
        for child in self.children: yield from iter(child) 
    def __reversed__(self):
        for child in self.children: yield from reversed(child)
        yield self        
    
    def __repr__(self): return "{}(key='{}')".format(self.__class__.__name__, self.key)  
    def __str__(self): return '{}'.format(self.key)
    
    def addchildren(self, *others): 
        for other in others: self.addchild(other)        
    def addchild(self, other):
        assert isinstance(other, type(self))
        self.__children.append(other)    
    
    def addparent(self, other):
        assert isinstance(other, type(self))
        other.addchild(self)     

 
class Tree(object):
    def __init__(self, key, name=None):
        self.__key = key
        self.__name = name
        self.__nodes = {}         

    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key
    
    def __getitem__(self, nodekey): return self.__nodes[nodekey]
    def __iter__(self): 
        for nodekey, node in self.__nodes.items(): yield nodekey, node
    
    def __repr__(self): 
        if self.name: return "{}(key='{}', name='{}')".format(self.__class__.__name__, self.key, self.name)
        else: return "{}(key='{}')".format(self.__class__.__name__, self.key)    
    def __str__(self): 
        namestr = '{} ("{}")'.format(self.name if self.name else self.__class__.__name__, self.key)
        jsonstr = json.dumps(self, sort_keys=False, indent=3, separators=(',', ' : '), default=str)  
        return ' '.join([namestr, jsonstr])
       
    def append(self, *nodes):
        assert all([isinstance(node, Node) for node in nodes])
        assert not any([nodekey in self.__nodes.keys() for nodekey in [node.key for node in nodes]])
        self.__nodes.update({node.key:node for node in nodes})
        
    def __iadd__(self, other):
        assert isinstance(other, type(self))
        self.append(*[node for nodekey, node in iter(other)])
        return self  
    

class Renderer(object):
    def __init__(self, style='double', extend=0):
        self.__branch = _STYLES[style]['branch'] + _STYLES[style]['branch'][-1] * extend
        self.__terminate = _STYLES[style]['terminate'] + _STYLES[style]['terminate'][-1] * extend
        self.__run = _STYLES[style]['run'] + _STYLES[style]['run'][-1] * extend
        self.__blank = _STYLES[style]['blank'] + _STYLES[style]['blank'][-1] * extend
    
    def __call__(self, root):
        assert isinstance(root, Node)
        rows = [str(row) for row in self.rowgenerator(root)]
        return '\n'.join(rows)

    def rowgenerator(self, node, layers=[]):
        assert isinstance(node, Node)
        lastchild = lambda i, imax: i == imax
        pre = lambda i, imax: self.__terminate if lastchild(i, imax) else self.__branch
        pads = lambda: ''.join([self.__blank if layer else self.__run for layer in layers])
        
        if not layers: yield node.key
        
        imax = len(node.children) - 1
        for i, child in zip(range(len(node.children)), node.children):
            yield ''.join([pads(), pre(i, imax) , child.key])
            yield from self.rowgenerator(child, layers=[*layers, lastchild(i, imax)])
            

    