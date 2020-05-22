# -*- coding: utf-8 -*-
"""
Created on Fri May 15 2020
@name:   Statistics Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigh
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ['MonteCarlo']
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


class MonteCarlo(object):
    @property
    def keys(self): return list(self.__histograms.keys())
    @property
    def variables(self): 
        variables = {}
        for histogram in self.__histograms.values(): variables.update(histogram.variables)
        return variables
       
    def __init__(self, **histograms):
        self.__histograms = ODict([(key, value) for key, value in histograms.items()])
        self.__correlationmatrix = np.zeros((len(histograms), len(histograms)))
        np.fill_diagonal(self.__correlationmatrix, 1)
     
    def __samplematrix(self, size, *args, method='cholesky', **kwargs):
        samplematrix = np.array([histogram(size) for histogram in self.__histograms.values()]) 
        if method == 'cholesky':
            correlation_matrix = cholesky(self.__correlationmatrix, lower=True)
        elif method == 'eigen':
            evals, evecs = eigh(self.__correlationmatrix)
            correlation_matrix = np.dot(evecs, np.diag(np.sqrt(evals)))
        else: raise ValueError(method)
        return np.dot(correlation_matrix, samplematrix) 
    
    def dataframe(self, size, *args, **kwargs):
        samplematrix = self.__samplematrix(size, *args, **kwargs)    
        sampletable = {key:list(values) for key, values in zip(self.keys, samplematrix)}
        return pd.DataFrame(sampletable)        
    
    def __call__(self, size, *args, **kwargs):
        for index, values in self.dataframe(size).iterrows():  
            yield index, values, self.variables    
    
    
    
    
    
    