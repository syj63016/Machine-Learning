# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:27:08 2021

Perceptron

@author: Yingjian Song
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
    def fit(self, X_train, Y_train):
        # insert 1 to 1st column as bias
        X_train = np.insert(X_train, 0, 1, axis = 1)
        n_samples, n_features = X_train.shape
        # initial weights
        self.weights = np.ones([n_features, 1])
        print((self.weights).shape)
        # initial loss
        self.loss = [];
        # begin training
        for i in range(self.n_iters):
            #compute distance to the hyperplane
            Y_train = np.reshape(Y_train, (-1,1))
            result = np.dot(X_train, self.weights) * Y_train
            gradient = np.zeros([n_features, 1])
#            print((self.weights).shape)
            for j in range(n_samples):
                loss_ = 0
                cnt = 0
                if result[j] < 0:
                    # compute loss
                    loss_ = loss_ - result[j]
                    # count mis-classification samples
                    cnt = cnt + 1
                    # compute gradient descent
                    gradient = np.reshape((-Y_train[j] * X_train[j, :].T),(-1,1)) + gradient
            if cnt!=0:
                gradient = gradient / cnt
                loss_  = loss_/cnt
            self.weights = self.weights - self.learning_rate * gradient
            self.loss.append(loss_)
        plt.plot(range(0,self.n_iters), self.loss, color='r',label='loss')
    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis = 1)
        self.X_test = X_test
        prediction = np.sign(np.dot(self.X_test, self.weights))
        return prediction
            
