# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:16:27 2021

Support Vector Machine

@author: Yingjian Song
"""
import numpy as np
import math

class SVM(object):
    
    def __init__(self, n_iters, penalty_term, X_train):
        self.n_iters = n_iters
        self.C = penalty_term
        self.X_train = X_train
        self.num_datas, self.num_features = self.X_train.shape[0], self.X_train.shape[1]
        #initialize langrange Multipliers
        self.alpha = np.zeros(self.num_datas)
        #initialize Weights
        self.W = np.zeros(self.num_features)
        #initialize Bias
        self.b = 0
        
    def calcKernel(self, trainDataMat, sigma):
        #Gaussian kernel Function for training data
        self.trainDataMat = trainDataMat
        self.sigma = sigma
        num_datas = trainDataMat.shape[0]
        k = [[0 for i in range(num_datas)] for j in range(num_datas)]
        for i in range(num_datas):
            X = self.trainDataMat[i, :]
            for j in range(i, num_datas):
                Z = self.trainDataMat[j, :]
                result = np.sum((X - Z) * (X - Z))
                result = np.exp(-1 * result / (2 * self.sigma**2))
                k[i][j] = result
                k[j][i] = result
        return k
    
    def calcSinglKernel(self, x1, x2, sigma):
        #Gaussian kernel Function for predicting new data
        self.sigma = sigma
        result = np.sum((x1 - x2) * (x1 - x2))
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        return np.exp(result)
    
    def g_x(self, data_id, k, alpha, Y_train, b):
        pred = 0
        for j in range(self.num_datas):
            if alpha[j] > 0:
                # calculate predict value of data1
                pred = pred + alpha[j] * Y_train[j] * k[j][data_id]
        pred = pred + b
        return pred
    
    def isKKT(self, alpha, k, Y_train, b, C, j):
        #check if data satisfies KKT
            if (Y_train[j] * self.g_x(j, k, alpha, Y_train, b) > 1) and (alpha[j] == 0):
                return True
            elif (Y_train[j] * self.g_x(j, k, alpha, Y_train, b) == 1) and (alpha[j] < C) and (alpha[j] > 0):
                return True
            elif (Y_train[j] * self.g_x(j, k, alpha, Y_train, b) < 1) and (alpha[j] == C):
                return True
            else:
                return False
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        #kernel calculations
        self.k = self.calcKernel(self.X_train, 10)
        # implement SMO
        alpha1_old = 0
        alpha2_old = 0
        b_old = 0
        data2_id_pre = -1
        alpha_unchanged_count = 0
        for i in range(self.n_iters):
            print("*****************")
            print("n_round:", i)
            
            # find alpha which violates KKT
            for j in range(self.num_datas):
                    if (self.isKKT(self.alpha, 
                                   self.k, Y_train, 
                                   self.b, self.C, j) == False):
                        data1_id = j
                        break
                    if (j == self.num_datas-1):
                        return self.alpha, self.b
            #calculate Error1 for data1
            g_x1 = self.g_x(data1_id, self.k, self.alpha, Y_train, self.b)
            E1 = g_x1 -Y_train[data1_id]
            print("E1:", E1)
            # find data2 which is the most different from data1
            delta_error = 0
            for j in range(self.num_datas):
                    #calculate Error2 for data2
                    g_x2 = self.g_x(j, self.k, self.alpha, Y_train, self.b)
                    temp_E2 = g_x2 -Y_train[j]
                    if math.fabs(E1-temp_E2) > delta_error:
                        delta_error = math.fabs(E1-temp_E2)
                        alpha2 = self.alpha[j]
                        E2 = temp_E2
                        data2_id = j
            print("E2:", E2)
            alpha1_old = self.alpha[data1_id]
            alpha2_old = alpha2
            # update alpha2
            lamda = self.k[data1_id][data1_id] + self.k[data2_id][data2_id] - 2 * self.k[data1_id][data2_id]
            if lamda == 0:
                lamda = lamda + 0.1
            alpha2 = alpha2_old + Y_train[data2_id] * (E1-E2) / lamda
            print("data1_id:", data1_id)
            print("data2_id:", data2_id)
            if data2_id == data2_id_pre:
                alpha_unchanged_count = alpha_unchanged_count + 1
            if alpha_unchanged_count > 2:
                break
            data2_id_pre = data2_id
            #calculate upper bound and lower bound for alpha2
            if Y_train[data1_id] == Y_train[data2_id]:
                Low = np.max([0, alpha1_old + alpha2_old - self.C])
                High = np.min([alpha1_old + alpha2_old, self.C])
            else:
                Low = np.max([0, alpha2_old - alpha1_old])
                High = np.min([self.C + alpha2_old - alpha1_old, self.C])
            # check if alpha2 satisfy KKT
            if alpha2 < Low:
                alpha2 = Low
            elif alpha2 > High:
                alpha2 = High
            else:
                alpha2 = alpha2
            print("alpha2:", alpha2)
            print("*****************")
            # update alpha1
            alpha1 = alpha1_old + (alpha2_old - alpha2) * Y_train[data1_id] * Y_train[data2_id]
            # update alpha
            self.alpha[data1_id] = alpha1
            self.alpha[data2_id] = alpha2
            # update bias
            if (alpha2 < self.C) and (alpha2 > 0):
                temp1 = (alpha1_old - alpha1) * Y_train[data1_id] * self.k[data1_id][data2_id]
                temp2 = (alpha2_old - alpha2) * Y_train[data2_id] * self.k[data2_id][data2_id]
                self.b = b_old - E2 + temp1 + temp2
            elif (alpha1 < self.C) and (alpha1 > 0):
                temp1 = (alpha1_old - alpha1) * Y_train[data1_id] * self.k[data1_id][data1_id]
                temp2 = (alpha2_old - alpha2) * Y_train[data2_id] * self.k[data2_id][data1_id]
                self.b = b_old - E1 + temp1 + temp2
            else:
                temp1 = (alpha1_old - alpha1) * Y_train[data1_id] * self.k[data1_id][data2_id]
                temp2 = (alpha2_old - alpha2) * Y_train[data2_id] * self.k[data2_id][data2_id]
                temp3 = (alpha1_old - alpha1) * Y_train[data1_id] * self.k[data1_id][data1_id]
                temp4 = (alpha2_old - alpha2) * Y_train[data2_id] * self.k[data2_id][data1_id]
                self.b = (2 *b_old - E2 + temp1 + temp2 - E1 + temp3 + temp4)/2
            
            b_old = self.b
        return self.alpha, self.b
    
    def predict(self, X_test):
        self.X_test = X_test
        num_data = self.X_test.shape[0]
        predictions = np.zeros(num_data)
        for i in range(num_data):
            for j in range(self.num_datas):
                predictions[i] = self.alpha[j] * self.Y_train[j] * self.calcSinglKernel(self.X_train[j,:], 
                                                                      self.X_test[i,:], 
                                                                      10) + predictions[i]
            predictions[i] = predictions[i] + self.b
            if predictions[i] >= 0:
                predictions[i] = 1
            elif predictions[i] < 0:
                predictions[i] = -1
        return predictions
        
