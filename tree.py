# -*- coding: utf-8 -*-
"""
Created on Weds Sept 4 2019
@name:   Tree Object
@author: Jack Kirby Cook

"""

from utilities.strings import uppercase

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['Node', 'TreeRenderer']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_STYLES = {'double':dict(branch="╠══", terminate="╚══", run="║  ", blank="   "),
           'single':dict(branch="├──", terminate="└──", run="│  ", blank="   "),
           'single_round':dict(branch="├──", terminate="╰──", run="│  ", blank="   ")}

_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)   


class Node(object):
    def __init__(self, key):
        self.__key = key
        self.__children = []
    
    @property
    def key(self): return self.__key    
    @property
    def children(self): return self.__children    
    @property
    def offspring(self): 
        generator = iter(self)
        next(generator)
        return [node for node in generator]
           
    def __repr__(self): return '{}({})'.format(self.__class__.__name__, self.__key)
    def __str__(self): return uppercase(self.__key, withops=True)
    def __hash__(self): return str(self)
    def __len__(self): return len(self.offspring)
       
    def addchildren(self, *others):
        assert all([isinstance(other, type(self)) for other in others])
        for other in others: self.__children.append(other)
        
    def addparent(self, other):
        assert isinstance(other, type(self))
        other.addchild(self)
        
    def __iter__(self):
        yield self
        for child in self.children: yield from iter(child)
 
    def __reversed__(self):
        for child in self.children: yield from reversed(child)
        yield self
  

class TreeRenderer(object):
    def __init__(self, style='double', extend=0):
        self.__branch = _STYLES[style]['branch'] + _STYLES[style]['branch'][-1] * extend
        self.__terminate = _STYLES[style]['terminate'] + _STYLES[style]['terminate'][-1] * extend
        self.__run = _STYLES[style]['run'] + _STYLES[style]['run'][-1] * extend
        self.__blank = _STYLES[style]['blank'] + _STYLES[style]['blank'][-1] * extend
    
    def __call__(self, root):
        rows = [row for row in self.rowgenerator(root)]
        return '\n'.join(rows)

    def rowgenerator(self, node, layers=[]):
        assert isinstance(node, Node)
        lastchild = lambda i, imax: i == imax
        pre = lambda i, imax: self.__terminate if lastchild(i, imax) else self.__branch
        pads = lambda: ''.join([self.__blank if layer else self.__run for layer in layers])
        
        if not layers: yield uppercase(node.key, withops=True)
        
        imax = len(node.children) - 1
        for i, child in zip(range(len(node.children)), node.children):
            yield ''.join([pads(), pre(i, imax) ,uppercase(child.key, withops=True)])
            yield from self.rowgenerator(child, layers=[*layers, lastchild(i, imax)])
            




    
    
    
    
    
    