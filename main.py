# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:17:52 2020

@author: nduc2
"""


import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import support_functions as sf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import SoftMax_Regression as sr
from typing import *

dataset = load_iris()
data = dataset.data
features = dataset.feature_names
features.append('Species')
target = dataset.target
df_data = np.concatenate((data,target.reshape(len(target),1)), axis = 1)
dataframe = pd.DataFrame(data = df_data, columns = [features])
dataframe.info()
print(dataframe.describe())

df = dataframe.sample(frac = 1)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1].to_numpy(dtype = 'int32')
scaled_X = sf.standardize_data(X.to_numpy())


#Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
                size = 24)

cmap = sns.cubehelix_palette(light=1, dark = 0.1,
                              hue = 0.5, as_cmap=True)

sns.set_context(font_scale=2)

# Pair grid set up
g = sns.PairGrid(df)

# Scatter plot on the upper triangle
g.map_upper(plt.scatter, s=10, color = 'red')

# Distribution on the diagonal
g.map_diag(sns.distplot, kde=False, color = 'red')

# Density Plot and Correlation coefficients on the lower triangle
g.map_lower(sns.kdeplot, cmap = cmap)
g.map_lower(corrfunc);


X_train, X_test, y_train, y_test = sf.partition(scaled_X,Y,0.2)

def kFold(X: np.ndarray, k: int = 3) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    '''
    Partition array to k parts of test size and train size
    k: int, default = 3.
    Exception is thrown if k is less than 2
    '''
    if k < 2:
        raise Exception('Must have at least 2 partitions. k = {}'.format(k))
    n = X.shape[0]
    
    if k > n:
        raise Exception('k has to be smaller or equal to n, {} > {}'.format(k, n))
    
    X_index = [i for i in range(n)]
    partitions = {}
    
    for i, test in enumerate(np.array_split(X_index, k)):
        train = np.setdiff1d(X_index, test)
        partitions[i] = (train, test)
        
    return partitions


def find_optimal_params(lambds: List[float], learning_rates: List[float], regularizers: List[str], cv: int, tol: List[float]) -> Dict[str, any]:
    partitions = kFold(scaled_X, cv)
    optimal_params = {'lambd': -1, 'learning_rate': -1, 'regularizer': None, 'cv': -1, 'score': -1, 'tol': -1}
    max_score = 0
    for lambd in lambds:
        for lr in learning_rates:
            for reg in regularizers:
                for t in tol:
                    for i, partition in partitions.items():
                        s_reg = sr.Softmax_Regression()
                        train, test = partition
                        X_train, X_test = scaled_X[train], scaled_X[test]
                        y_train, y_test = Y[train], Y[test]
                        s_reg.fit(X_train, y_train, learning_rate=lr, lambd=lambd, regularizer=reg, tol = t)
                        y_test_predict = s_reg.predict(X_test)
                        score,error = sf.accuracy_generalized_error(y_test, y_test_predict)
                        if score > max_score:
                            max_score = score
                            optimal_params = {'lambd': lambd, 'learning_rate': lr, 'regularizer': reg, 'cv_index': i, 'score': max_score, 'tol' : t}
                        
    return optimal_params

lambds = [0.1,0.01, 0.001, 0.0001]
learning_rates = [0.1, 0.01, 0.001]
regularizers = ['l1', 'l2', None]  
tol =  [0.001, 0.0001, 0.00001, 0.000001]
optimal_params = find_optimal_params(lambds = lambds, learning_rates = learning_rates, regularizers = regularizers, cv = 5, tol = tol)
print('Optimal_params: ', optimal_params)

