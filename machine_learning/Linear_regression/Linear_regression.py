# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:14:49 2021

Linear Regression algorithm

@author: Yingjian Song
"""
import numpy as np
import matplotlib.pyplot as plt

class Linear_regression(object):
    def __init__ (self, learning_rate, n_iters, regularization = None, regularization_lr = 0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.regularization_lr = regularization_lr
    def fit(self, X_train, Y_train):
        X_train = np.insert(X_train, 0, 1, axis = 1)
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features);
#        self.bias = 0;
        self.loss = [];
        for i in range(self.n_iters):
#            self.diff = np.dot(X_train, self.weights) + self.bias - Y_train
            self.diff = np.dot(X_train, self.weights) - Y_train
            if self.regularization == None:
                gradient = (np.dot((self.diff).T, X_train)) / n_samples
            elif (self.regularization == 'l1'):
                gradient = (np.dot((self.diff).T, X_train) + self.regularization_lr * np.sign(self.weights)) / n_samples
            elif (self.regularization == 'l2'):
                gradient = (np.dot((self.diff).T, X_train) + self.regularization_lr * self.weights) / n_samples
            self.weights = self.weights - gradient.T
#            self.bias = self.bias - (sum((self.diff))) / n_samples
            self.loss.append(np.mean(np.power(self.diff, 2)))
        plt.plot(range(0,self.n_iters)[-100 : ], self.loss[-100 : ], color='r',label='loss')
    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis = 1)
#        prediction = np.dot(X_test, self.weights) + self.bias
        prediction = np.dot(X_test, self.weights)
        return prediction
