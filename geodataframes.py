# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 2018
@name:   GeoDataFrame Functions
@author: Jack Kirby Cook

"""

import os.path
import geopandas as gp

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['geodataframe_fromshapefile', 'geodataframe_fromdir', 'geodataframe_fromfile']
__copyright__ = "Copyright 2018, Jack Kirby Cook"
__license__ = ""


_aslist = lambda items: [items] if not isinstance(items, (list, tuple)) else list(items)
_filterempty = lambda items: [item for item in _aslist(items) if item]


# FACTORY
_geoids = lambda geography: [geography[slice(0, i+1)].geoID for i in reversed(range(len(geography)))]
_shapedirname = lambda shape, geoid, year: '_'.join(_filterempty([shape, geoid.replace('X', ''), str(year)]))
_shapedirnames = lambda shape, geography, year: [_shapedirname(shape, geoid, year) for geoid in _geoids(geography)] + [_shapedirname(shape, '', year)]

class ShapeFileMissingError(Exception):
    def __init__(self, shapeID, geoid, year): 
        super().__init__('shape={}, geography={}, year={}'.format(shapeID, geoid, year))   
    
def geodataframe_fromshapefile(shape, *args, geography, year, directory, **kwargs):
    for shape_dirname in _shapedirnames(shape, geography, year):
        shape_repository = os.path.join(directory, shape_dirname)
        if os.path.isdir(shape_repository): break
        else: shape_repository = None
    if not shape_repository: raise ShapeFileMissingError(shape, geography.geoid, year)   
    return geodataframe_fromdir(shape_repository)
    

# FILE
def geodataframe_fromdir(directory):
    try:
        for file in os.listdir(directory):
            if file.endswith(".shp"): geodataframe = gp.read_file(os.path.join(directory, file))
        print('File Loading Success:')
        print(str(directory), '\n')   
    except Exception as error:
        print('File Loading Failure:')
        print(str(directory), '\n')         
        raise error
    return geodataframe
        
def geodataframe_fromfile(file):
    try:
        geodataframe = gp.read_file(file)
        print('File Loading Success:')
        print(str(file), '\n')   
    except Exception as error:
        print('File Loading Failure:')
        print(str(file), '\n')         
        raise error
    return geodataframe








