# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:52:46 2021

Independent Component Analysis

this code assumes that numbers of observation signals equals to the number of 
source signals

@author: Yingjian Song
"""

import numpy as np

class ICA():
    
    def __init__(self, n_components, error_tolerance, learning_rate):
        '''
        Parameters
        ----------
        n_components : int
            number of independent componets you wish to recover.
        error_tolerance : float
            the error you can tolerance to convergence during gradient descent.
        learning_rate : float
            learning rate for gradient descent algorithm
        
        Returns
        -------
        None.

        '''
        self.n_components = n_components
        self.error_tolerance = error_tolerance
        self.learning_rate = learning_rate
        self.W = np.ones((self.n_components, self.n_components))

    def sigmoid(self, s):
        '''
        choose sigmoid function as Cumulative Distribution Function of Source signals

        '''
        return 1/(1+np.exp(-s))
    
    def decorrelation(self, W):
        eigenvalue, eigenvector = np.linalg.eigh(np.dot(W, W.T))
        for i in range(len(eigenvalue)):
            if eigenvalue[i] <= 0.0000001:
                eigenvalue[i] = eigenvalue[i] + 0.0000001
        return np.dot(np.dot(eigenvector * (1. / np.sqrt(np.abs(eigenvalue))), eigenvector.T), W)
        
    def preprocessing(self, X):
        '''        
        This Function performs centering and whitening on Observed Data X
        
        
        Parameters
        ----------
        X : matrix
            N dimensional matrix, Oberserved data. Each row represents one observed signal
        Returns
        -------
        X:
            normalized X

        '''
        
        self.X = X
        if len(X.shape) == 1:
            self.X = np.reshape(self.X, (1,-1))
        
        # centering data
        self.X = self.X-np.reshape(np.mean(self.X, axis = 1), (-1,1))
        
        # whitening data
        # get coviriance matrix
        if self.X.shape[0] == 1:
            cov_X = np.dot(self.X, self.X.T)/(self.X.shape[1]-1)
        else:
            cov_X = np.cov(self.X)

        #compute eigenvalue, eigenvector 
        eigenvalue, eigenvector = np.linalg.eig(cov_X)
        for i in range(len(eigenvalue)):
            if eigenvalue[i] <= 0.0000001:
                eigenvalue[i] = eigenvalue[i] + 0.0000001
        #calculate -1/2 power of eigenvalue
        eigenvalue = np.sqrt(1/eigenvalue)
        eigenvalue = np.diag(eigenvalue)
        # calculate transform matirx
        v = np.dot(eigenvalue, eigenvector.T)
        
        #perform whitening transform
        self.X = np.dot(v, self.X)
        return self.X
    
    def fastICA(self, X):
        # preprocessing observed data
        self.X = self.preprocessing(X)
        temp = np.zeros((self.n_components, self.n_components))
        difference = 1
        # update weights by SGD
        while (difference>self.error_tolerance):
            for i in range(self.n_components):
                s_i = np.dot(self.W[i, :], self.X)
                temp[i,:] = np.dot((1-2*self.sigmoid(s_i)), self.X.T)
            #Calculate Gradient
            gradient = temp + np.linalg.pinv(self.W.T)
            W_pre = self.W
            # update weights
            self.W = self.W + self.learning_rate * gradient
            # decorrelate Weights
            self.W = self.decorrelation(self.W)
            # check if convergence
            difference = np.max( np.abs(np.abs(np.diag(np.dot(self.W, W_pre.T))) - 1) )
        #recover signals
        self.S = np.dot(self.W, self.X)
        return self.W, self.X, self.S
        