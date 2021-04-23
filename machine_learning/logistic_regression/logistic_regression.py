# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:37:46 2021

@author: Yingjian Song
"""

import numpy as np

def sigmoid(z):
    result = 1 / (np.exp(-z) + 1)
    return result

class logistic_regression(object):
    def __init__ (self, n_iters, lr):
        self.n_iters = n_iters
        self.lr = lr
        
    def fit(self, X_train, Y_train):
        X_train = np.insert(X_train, 0, 1, axis = 1)
        Y_train = np.reshape(Y_train, (-1,1))
        self.X_train = X_train
        self.Y_train = Y_train
        self.loss = []
        self.W = np.zeros((self.X_train.shape[1], 1))
        for i in range(self.n_iters):
            # compute MAP
            obj_function = Y_train * np.log(sigmoid(np.dot(self.X_train, self.W)) + 0.0000001) + (1-self.Y_train) * np.log(1-sigmoid(np.dot(self.X_train, self.W)) + 0.0000001)
            #compute loss
            loss_temp = np.sum((Y_train - obj_function)**2)
            self.loss.append(loss_temp)
            # compute gradient
            gradient = -np.dot((Y_train - sigmoid(np.dot(self.X_train, self.W))).T, self.X_train).T
            # print(gradient)
            # gradient descent
            self.W = self.W - self.lr * gradient
    
    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis = 1)
        predictions = sigmoid(np.dot(X_test, self.W))
        for i in range(len(predictions)):
            predictions[i] = [1 if predictions[i] > 0.5 else 0] 
        return predictions, self.W, self.loss
