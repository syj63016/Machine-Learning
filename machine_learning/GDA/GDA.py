# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:11:01 2021

@author: Administrator
"""
import numpy as np

def Gussian(mean, covariance, x):
    if x is np.array:
        x = np.reshape(x, (-1,1))
        mean = np.reshape(mean, (-1,1))
        exp_part =  np.dot (np.dot((x-mean).T , (np.linalg.inv(covariance))),  (x-mean))
        res = (1/ (((2*np.pi) ** (len(x)/2)) * (np.linalg.det(covariance) ** (1/2)))) * np.exp(-1/2 *exp_part)
    else:
        exp_part = -1/2 * ((x-mean)**2/ covariance)
        res = 1/(np.sqrt(2 * np.pi * covariance)) * np.exp(-1/2 *exp_part)
    return res

class Gussian_discriminant_analysis():
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
        #calculate mean of classes:
        self.mean_1 = np.sum(self.X_train[Y_train == 1], axis = 0) / len(self.Y_train[Y_train == 1])
        self.mean_0 = np.sum(self.X_train[Y_train == 0], axis = 0) / len(self.Y_train[Y_train == 1])
        #calculate covariance of classes:
        self.covariance = np.dot((self.X_train[Y_train == 1] - self.mean_1).T, (self.X_train[Y_train == 1] - self.mean_1))
        
        return self.mean_1, self.mean_0, self.covariance
    
    def predict(self, X_test):
        self.X_test = X_test
        predictions  = np.zeros((self.X_test.shape[0], 1))
        for i in range(X_test.shape[0]):
            
            #calculate probility of data label equals to 1:        
            prob1 = Gussian(self.mean_1, self.covariance, self.X_test[i,:])
            
            #calculate probility of data label equals to 0:
            prob0 = Gussian(self.mean_0, self.covariance, self.X_test[i,:])
            
            # get prediction
            if prob1 > prob0:
                predictions[i] = 1
            else:
                predictions[i] = 0
                
        return predictions
        
        