# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:17:21 2021

@author: Administrator
"""

'''
Student A B C D E F
Midterm Grade 92 55 100 88 61 75
Course Grade 95 70 95 85 75 80
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

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

Data = {
        'Student': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Midterm Grade': [92, 55, 100, 88, 61, 75],
        'Course Grade': [95, 70, 95, 85, 75, 80]
        }

X = Data['Midterm Grade']
X = np.array(X)
X = np.reshape(X,(-1,1))
X_raw_mean = np.mean(X)
X_raw_std = np.std(X)
X_normalized = (X-X_raw_mean)/X_raw_std
X_normalized = np.insert(X_normalized,0,1, axis = 1)

Y = Data['Course Grade']
Y = np.array(Y)
Y = np.reshape(Y,(-1,1))
Y_raw_mean = np.mean(Y)
Y_raw_std = np.std(Y)
Y_normalized = (Y-Y_raw_mean)/Y_raw_std

step_length = 0.001
n_iters = 10000
W, loss, y_preds = train_function(step_length, X_normalized, Y_normalized, n_iters)

#unnormalized W
Y_pre_unnormalized = y_preds * Y_raw_std + Y_raw_mean

plt.scatter(X, Y)
plt.plot(X, Y_pre_unnormalized)
plt.show()