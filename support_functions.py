# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:30:39 2020

@author: nduc2
"""

import numpy as np

def accuracy_generalized_error(u,v):
    accuracy = np.mean(u==v)
    err = 1-accuracy
    return accuracy, err

def one_hot_labels(Y):
    num_col = len(np.unique(Y))
    result = np.zeros((len(Y),num_col))
    for i in range(len(Y)):
        result[i][Y[i]] = 1
        
    return result

def softmax_score(score):
    softmax_score = np.exp(score)/np.sum(np.exp(score))
    return softmax_score

def cross_entropy_loss(y_one_hot, y_proba):
    # n = y_one_hot.shape[1]
    loss = y_one_hot * np.log(y_proba)
    loss = np.sum(loss)
    return loss

def standardize_data(X):
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        X[:,i] = np.array(X[:,i])
        std = np.std(X[:,i])
        X[:,i] = (X[:,i] - mean) / std
        
    return X

#Partition data to train set and test set
def partition(X,y,t):
   mark = int(t*(len(X)-1))
   X_train = X[:-mark,:]
   X_test = X[len(X)-mark:,:]
   y_train = y[:-mark]
   y_test = y[len(X)-mark:]
   
   return X_train, X_test, y_train, y_test

# from sklearn.metrics import log_loss

# a = np.array([2.0, 1.0, 0.1])

# b = np.array([[0.3,0.4,0.8],[0.2,0.3,0.7],[0.8,0.4,0.3],[0.2,0.6,0.9],[0.2,0.7,0.4]])
# print(log_loss(a,b, labels = [0,1,2]))

# print(cross_entropy_loss(one_hot_labels(a),b))