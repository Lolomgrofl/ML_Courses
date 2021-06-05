# -*- coding: utf-8 -*-
"""
@author: Sandro Radovanović
"""

# import pandas as pd
import numpy as np

# from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB

# ---------- RANDOM FOREST ----------
class RandomForestClassification:
    
    def __init__(self, n_estimators = 100, random_state = 2021):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        return
        
    def param_opt(self):
        param_grid = {
                'n_estimators': [range(100, 500, 50)],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        return param_grid
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X, decision_threshold=0.5):
        predictions = self.model.predict_proba(X) >= decision_threshold
        
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def feature_importance(self):
        return self.model.feature_importances_
    
# ---------- GRADIENT BOOSTING ----------
class GradientBoostingClassification:
    
    def __init__(self, n_estimators = 100):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators)
        return
    
    def param_opt(self):
        param_grid = {
                'n_estimators': [range(100, 500, 50)],
            }
        
        return param_grid
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X, decision_threshold=0.5):
        predictions = self.model.predict_proba(X) >= decision_threshold
        
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def feature_importance(self):
        return self.model.feature_importances_
    
# ---------- LOGISTIC REGRESSION ----------
class LogisticRegressionClassification:
    
    def __init__(self, C = 1):
        self.model = LogisticRegression(C=C)
        return
    
    def param_opt(self):
        param_grid = {
                'C': np.linspace(0.0001, 2, 20),
                'penalty' : ['l1', 'l2', 'elasticnet']
            }
        
        return param_grid
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X, decision_threshold=0.5):
        predictions = self.model.predict_proba(X) >= decision_threshold
        
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def feature_importance(self):
        return self.model.coef_