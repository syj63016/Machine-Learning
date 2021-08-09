# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:04:50 2021

@author: yingjian song
"""

import numpy as np

class ICA():
    
    def __init__(self, n_components, error_tolerance, learning_rate, max_iteration):
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
        self.W = np.random.rand(self.n_components, self.n_components)
        self.max_iteration = max_iteration
    
    def norm(self, x):
        """
        Parameters
        ----------
        x : vector
        
        Returns
        -------
        norm_x : float
            norm of input vector.

        """
        for i in range(len(x)):
            if i == 0:
                norm_square = x[i] * x[i]
            else:
                norm_square = x[i] * x[i] + norm_square
        norm_x = pow(norm_square, 0.5)
        return norm_x

    def filter_matrix(self, X):
        """
        Parameters
        ----------
        X : 2D matrix

        Returns
        -------
        X : matrix 
        filter elements which are smaller than 1e-07 to 0

        """
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j] < 1e-07:
                    X[i,j] = 0
        return X

    def orthonormal(self, Q):
        for i in range(1, Q.shape[0]):
            if i == 1:
                Q[i-1, :] = Q[i-1, :] / self.norm(Q[i-1, :])
            for j in range(i):
                inner_p = np.sum(Q[j, :] * Q[i, :])
                if j == 0:
                    plane = inner_p * Q[j, :]
                else:
                    plane = plane + inner_p * Q[j, :]
            Q[i, :] = Q[i, :] - plane
            Q[i, :] = Q[i, :] / self.norm(Q[i, :])
        return Q

    def sigmoid(self, s):
        '''
        choose sigmoid function as Cumulative Distribution Function of Source signals
        '''
        for i in range(len(s)):
            if s[i]>=0:
                s[i] = 1/(1+np.exp(-s[i]))
            else:
                s[i] = np.exp(s[i]) / (1 + np.exp(s[i]))
        return s
    
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def d_tanh(self, x):
        res = 1-self.tanh(x) ** 2
        return res
    
    def decorrelation(self, W):
         #W <- (W * W.T) ^{-1/2} * W
        eigenvalue, eigenvector = np.linalg.eigh(np.dot(W, W.T))
        # for i in range(len(eigenvalue)):
        #     if eigenvalue[i] <= 0.0000001:
        #         eigenvalue[i] = eigenvalue[i] + 0.0000001
        return np.dot(np.dot(eigenvector * (1. / np.sqrt(eigenvalue)), eigenvector.T), W)
        # s, u = np.linalg.eigh(np.dot(self.W, self.W.T))
        # return np.linalg.multi_dot([u * (1. / np.sqrt(s)), u.T, self.W])
    
    def center(self, X):
        '''        
        This Function performs centering on Observed Data X
        
        
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
        return self.X
    
    def Whiten(self, X):
        '''        
        This Function performs whitening on Observed Data X
        
        
        Parameters
        ----------
        X : matrix
            N dimensional matrix, Oberserved data. Each row represents one observed signal
        Returns
        -------
        X:
            normalized X
        '''
        # # get coviriance matrix
        if self.X.shape[0] == 1:
            cov_X = np.dot(self.X, self.X.T)/(self.X.shape[1]-1)
        else:
            cov_X = np.cov(self.X)
        eigenvalue, eigenvector = np.linalg.eig(cov_X)
        self.X = np.dot(np.dot(eigenvector * (1. / np.sqrt(eigenvalue)), eigenvector.T), self.X)
        return self.X
    
    def fastICA(self, X):
        # preprocessing observed data
        self.X = self.center(X)
        self.X = self.Whiten(X)
        temp = np.zeros(self.n_components)
        n_iterations = 0
        self.W = self.decorrelation(self.W)
        W_pre = self.W
        # update weights by Newton Method
        while (n_iterations < self.max_iteration):
            n_iterations = n_iterations + 1
            for i in range(self.n_components):
                temp[i] = self.d_tanh(np.dot(self.W[i], self.X)).mean()
            self.W = np.dot(self.tanh(np.dot(self.W, self.X)), self.X.T) / self.X.shape[1] - temp[:, np.newaxis] * self.W
            self.W = self.decorrelation(self.W)
            # check if convergence
            difference = np.max( np.abs(np.abs(np.diag(np.dot(self.W, W_pre.T))) - 1) )
            W_pre = self.W
            print(n_iterations)
            if difference < self.error_tolerance:
                break
        #recover signals
        self.S = np.dot(self.W, self.X)
        # self.S = np.linalg.multi_dot([self.W, self.K, self.X])
        return self.W, self.X, self.S