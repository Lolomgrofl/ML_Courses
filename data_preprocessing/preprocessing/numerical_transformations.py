# -*- coding: utf-8 -*-
"""
@author: Sandro Radovanović
"""

import pandas as pd
import numpy as np
import scipy.stats as stats


class LogTransformNumerical:
    
    def __init__(self):
        return
    
    def fit(self, X, threshold = 3):
        X_num = X.select_dtypes(np.number)
        skewness = stats.skew(X_num) >= threshold
        
        self.columns = X_num.columns[skewness].values
        
        return
    
    def transform(self, X):
        X_n = X.copy()
        
        X_n[self.columns] = np.log1p(X_n[self.columns])
        
        return X_n