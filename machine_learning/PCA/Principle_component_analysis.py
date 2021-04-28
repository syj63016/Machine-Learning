# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:26:05 2021

Principle component analysis in 'max projection variance view'

@author: Yingjian Song
"""
import numpy as np

class Principle_Component_Analysis():
    
    def fit(self, X):
        '''
        Parameters
        ----------
        X : 2-D array
            Input size of X is N * P, which N is number of samples and P is
            number of features.
        Returns
        -------
        New_space : 2-D array
                    new feature space in 'max projection variance'
        '''
        self.X = X
        mean = np.mean(X, axis = 0)
        
        # center the data along each feature
        X_centered = self.X - mean
        print(np.mean(X_centered, axis = 0))
        # get convirance matrix
        cov_x = np.dot(X_centered.T, X_centered)
        
        #calculate eigenvalue and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_x)
        
        #normalize eigenvectors
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis = 0)
        
        # sort eigenvalues and based on sorted eigenvalues
        eigen_pairs = []
        for i in range(len(eigenvalues)):
            eigen_pairs.append( (np.abs(eigenvalues[i]), eigenvectors[:,i]) )
        
        eigen_pairs = sorted(eigen_pairs, key= lambda x : x[0], reverse = True)
        
        self.New_space = []
        for i in range(len(eigenvalues)):
            self.New_space.append(eigen_pairs[i][1])
        self.New_space = np.array(self.New_space).T
        
        return self.New_space
    
    def transform(self, X, K):
        '''
        Parameters
        ----------
        X : 2-D array
            Input size of X is N * P, which N is number of samples and P is
            number of features.
        K : int
            number of features to keep which should be less than P.

        Returns
        -------
        transformed_X: 2-D array
                       transformed matrix in new feature space,
                       size of transformed_X should be N * K
        '''
        self.X = X
        # transform into most projection variance direction
        transformed_X = np.dot(self.X, self.New_space[:, :K])
        
        return transformed_X