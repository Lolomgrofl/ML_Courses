# -*- coding: utf-8 -*-
"""
@author: Sandro Radovanović
"""

import pandas as pd
import numpy as np
import scipy.stats as stats


import logging
logging.basicConfig(filename='../log_file.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def accuracy(y, y_hat):
    '''
    Calculate classification accuracy
        
    Parameters:
    y: numpy.array
        Vector of true values
    y_hat: numpy.array
        Vector of predicted values
    
    Output:
    decimal: Accuracy score
    '''
    
    try:
        return np.sum(y == y_hat)/len(y)
        
    except Exception as e:
        logging.error('Accuracy score error: ' + e.msg)
    
def precision(y, y_hat):
    '''
    Calculate classification precision score
        
    Parameters:
    y: numpy.array
        Vector of true values
    y_hat: numpy.array
        Vector of predicted values
    '''
    
    try:
        return np.sum((y == y_hat)[y_hat == 1])/np.sum(y_hat == 1)
        
    except Exception as e:
        logging.error('Precision score error: ' + e.msg)
            
def recall(y, y_hat):
    '''
    Calculate classification recall score
        
    Parameters:
    y: numpy.array
        Vector of true values
    y_hat: numpy.array
        Vector of predicted values
    '''
    
    try:
        return np.sum((y == y_hat)[y == 1])/np.sum(y == 1)
        
    except Exception as e:
        logging.error(e.msg)
            
def f1_score(y, y_hat):
    '''
    Calculate classification F1 score
        
    Parameters:
    y: numpy.array
        Vector of true values
    y_hat: numpy.array
        Vector of predicted values
    '''
    try:
        prec = precision(y, y_hat)
        rec = recall(y, y_hat)
            
        return 2*prec*rec/(prec + rec)
        
    except Exception as e:
        logging.error(e.msg)
            
def auc(y, y_hat):
    '''
    Calculate AUC score
        
    Parameters:
    y: numpy.array
        Vector of true values
    y_hat: numpy.array
        Vector of probabilities for the desired outcome
    '''
    try:
        n1 = np.sum(y == 1)
        n2 = np.sum(y == 0)
            
        R1 = np.sum(stats.rankdata(y_hat)[y == 1])
        U = R1 - (n1*(n1+1))/2
            
        return (U/(n1*n2))
        
    except Exception as e:
        logging.error(e.msg)