# -*- coding: utf-8 -*-
"""
@author: Sandro Radovanović
"""

import numpy as np
import pandas as pd

class DummyCoding:
    
    def __init__(self):
        return
    
    def fit(self, X, columns):
        self.columns = columns
        self.values = [X[x].unique().tolist() for x in self.columns]
        
    def transform(self, X, append=True):
        df = X.copy()
        if append == False:
            df = df.drop(df.columns, axis=1)
            
        for i in range(len(self.columns)):
            for j in self.values[i]:
                df[f'{self.columns[i]}_{j}'] = (X[f'{self.columns[i]}'] == j).astype(int)
        
        df = df.drop(self.columns, axis=1)
        return df