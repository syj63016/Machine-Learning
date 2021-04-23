# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:57:17 2021

@author: Yingjian Song
"""
import numpy as np

class Linear_Discriminant_Analysis(object):
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        y_classes = []
        for i in y_train:
         if i not in y_classes:
              y_classes.append(i)
        class_mean_vector = []
        # compute mean for each classes
        for i in y_classes:
            class_mean_vector.append(np.mean(X_train[y_train == i], axis = 0))
        # compute within-class matrix
        s_w = np.zeros(X_train.shape[1], X_train.shape[1])
        for i in range(len(y_classes)):
            temp = X_train[y_train == y_classes[i]] - class_mean_vector[i]
            n_samples_for_each_class = temp.shape[0]
            s_w = s_w + (np.dot(np.mat(temp).T, np.mat(temp))/n_samples_for_each_class)
        # compute between-class matrix
        s_b = np.zeros(X_train.shape[1], X_train.shape[1])
        for i in range(len(y_classes)):
            overall_mean = np.mean(X_train, axis = 0)
            temp_mean = np.mat(class_mean_vector[i] - overall_mean[i])
            s_b = s_b + np.dot(temp_mean.T, temp_mean)
        # compute eigenvalue
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(s_w) * s_b)
        # sort eigenvector based on eigenvalues
        eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        # find new feature space
        self.W = np.zeros((eigvecs.shape))
        for i in range(len(eig_pairs)):
            self.W[:,i] = eig_pairs[i][1]
        
    def transform(self, X_test):
        self.X_test = X_test
        predictons = np.dot(self.X_test, self.W)
        return predictons
        
            
