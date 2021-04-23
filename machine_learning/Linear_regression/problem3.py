# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:30:26 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

def train_function(step_length, X, Y, n_iters, W = None):
    if W is None:
        W = np.zeros((X.shape[1],1))
    loss =[]
    for i in range(n_iters):
        n_samples = X.shape[0]
        y_pred = np.dot(X, W)
        gradient = np.dot((y_pred - Y).T, X)/n_samples
        W = W - step_length * gradient.T
        loss_temp = sum((y_pred - Y)**2)/n_samples
        loss.append(loss_temp)
    y_preds = np.dot(X, W)
    return W, loss, y_preds
# construct data dict
Data = {
        'Person': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'Weight': [69, 83, 77, 75, 71, 73, 67, 71, 77, 69, 74, 86, 84],
        'Age': [50, 20, 20, 30, 30, 50, 60, 50, 40, 55, 40, 40, 20],
        'Stress': [55, 47, 33, 65, 47, 58, 46, 68, 70, 42, 33, 55, 48],
        'BP':[120, 141, 124, 126, 117, 129, 123, 125, 132, 123, 132, 155, 147]
        }
## construct X and Y
X = []
Y = []
for key in Data.keys():
    if key != 'BP' and key != 'Person':
        X.append(Data[key])
    elif key == 'BP':
        Y.append(Data[key])

# Normalize X and Y
X = np.array(X).T
X_normalized = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
X_normalized = np.insert(X_normalized,0,1, axis = 1)

Y = np.array(Y).T
Y_raw_mean = np.mean(Y, axis = 0)
Y_raw_std = np.std(Y, axis = 0)
Y_normalized = (Y - Y_raw_mean) / Y_raw_std

# set step_length and number of iterations
step_length = 0.001
n_iters = 50000

# start training 
W, loss, y_preds = train_function(step_length, X_normalized, Y_normalized, n_iters)
# unnormalize predictions
Y_pre_unnormalized = y_preds * Y_raw_std + Y_raw_mean
plt.scatter(Y, Y_pre_unnormalized)
plt.plot(Y, Y, c = 'orange')
plt.show()