# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:54:40 2020

@author: nduc2
"""

import support_functions as sf
import numpy as np

class Softmax_Regression:
    
    def __init__(self):
        pass
    
    def compute_score(self, w, X):
        num_sample = X.shape[0]
        num_class = w.shape[1]
        num_features = X.shape[1]
        
        score = np.zeros((len(X), num_class))
        
        for i in range(num_sample):
            
            for j in range(num_class):
                score[i][j] = np.dot(np.matrix(w[:,j]), np.matrix(X[i]).T) 
                
        return score
            
    
    def fit(self,X,y,learning_rate = 0.01, epochs = 1000, tol = None, regularizer = None,
            lambd = 0.0, early_stopping = False, validation_fraction = 0.1, **kwargs):
        num_row = X.shape[0]
        num_features = X.shape[1]
        num_class = len(np.unique(y))
        w = np.empty((num_features+1,num_class))
        ones = np.ones((1,len(X)))
        X = np.concatenate((X,ones.T),axis = 1)
        y_one_hot = sf.one_hot_labels(y)
        prev_cost = 0
        cost = 0
        for i in range(epochs):
            print('Epoch:' , i+1)
            score = self.compute_score(w,X)
            y_proba = sf.softmax_score(score)
            error = y_proba-y_one_hot
            derivative_nll = np.dot(X.T,error)
            if regularizer == None:
                w = w - (learning_rate/num_row)*derivative_nll
                cost = sf.cross_entropy_loss(y_one_hot,y_proba)/num_row
            elif regularizer == 'l2':
                w = w - (learning_rate/num_row)*(derivative_nll + lambd*w)         
                cost = (sf.cross_entropy_loss(y_one_hot,y_proba)/num_row) + (lambd/2)*np.sum(w**2)
            elif regularizer == 'l1':
                print('l1')
                w = w = w - (learning_rate/num_row)*(derivative_nll + lambd*np.sign(w))
                cost = (sf.cross_entropy_loss(y_one_hot,y_proba)/num_row) + (lambd/2)*np.sum(w**2)
            
            print('Cost :', cost)
            if abs(cost - prev_cost) < tol:
                self.w = w
                print('break')
                break
            else:
                prev_cost = cost
        
        self.w = w
        
        return self.w
        
        
    def predict(self, X):
        y_pred = np.zeros(len(X))
        ones = np.ones((1,len(X)))
        X = np.concatenate((X,ones.T), axis = 1)
        score = np.dot(X, self.w)
        score = sf.softmax_score(score)
        
        for i in range(len(X)):
            y_pred[i] = np.argmax(score[i])
            
        return y_pred
        