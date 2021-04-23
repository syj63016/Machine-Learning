# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:36:23 2021

Naive Bayes
@author: syj
"""
import numpy as np

def likelihood(x, mean, variance):
    numerator = np.exp(-(x-mean)**2/(2*variance))
    denominator = 1/np.sqrt(2*np.pi*variance)
    class_condition_prob = numerator * denominator
    return class_condition_prob

class Naive_Bayes():
    
    def fit(self, X_train, Y_train, is_feature_discrete):    
        self.X_train = X_train
        self.Y_train = Y_train
        self.is_feature_discrete = is_feature_discrete
        num_samples, num_features = self.X_train.shape
        self.num_classes = len(np.unique(self.Y_train))
        self.classes_prior = np.zeros(self.num_classes)
        
        # calculate prior for each class
        for i in range(self.num_classes):
            self.classes_prior[i] = len(self.Y_train[self.Y_train == np.unique(self.Y_train)[i]]) / num_samples
            
        ### calculate likehood for each class
        # count discrete features
        self.likehoods_for_all_features = []
        # n_classes_within_feature_for_all_discrete_feature = []
        for i in range(num_features):
            if self.is_feature_discrete[i] == 1:
                # get number of classes in each feature
                classes_within_feature = np.unique(self.X_train[:,i])
                num_classes_within_feature = len(classes_within_feature)
                temp = np.zeros(self.num_classes + 1, num_classes_within_feature)
                temp[-1,:] = classes_within_feature
                # calculate likelihood for discrete features
                for k in range(self.num_classes):
                    for j in range(num_classes_within_feature):
                        temp_1 = self.X_train[:,i][self.Y_train == np.unique(self.Y_train)[k]]
                        temp[k, j] = (len(temp_1[temp_1 == classes_within_feature[j]]) + 1)/len(self.X_train[:,i][self.Y_train == np.unique(self.Y_train)[k]])
                self.likehoods_for_all_features.append(temp)
                
            else:
                # calculate mean and variance
                temp = np.zeros((self.num_classes, 2))
                for k in range(self.num_classes):
                    temp[k,0] = np.mean(self.X_train[:,i][self.Y_train == np.unique(self.Y_train)[k]])
                    temp[k,1] = np.std(self.X_train[:,i][self.Y_train == np.unique(self.Y_train)[k]]) ** 2
                self.likehoods_for_all_features.append(temp)
        return self.likehoods_for_all_features
    
    def predict(self, X_test):
        self.X_test = X_test
        num_samples, num_features = self.X_test.shape
        y_pred = np.zeros(num_samples)
        for index in range(self.X_test.shape[0]):
            temp_y_pred = np.zeros(self.num_classes)
            for j in range(self.num_classes):
                #compute features likelihood
                for i in range(num_features):
                    if self.is_feature_discrete[i] == 1:
                        #compute discrete features likelihood
                        likelihood_feature = self.likehoods_for_all_features[i]
                        temp_y_pred[j] = temp_y_pred[j] + np.log(likelihood_feature[j, likelihood_feature[-1,:] == self.X_test[index, i]])
                    else:
                        #compute continues features likelihood
                        likelihood_feature = likelihood(self.X_test[index, i], 
                                                        self.likehoods_for_all_features[i][j, 0], 
                                                        self.likehoods_for_all_features[i][j, 1])
                        temp_y_pred[j] = temp_y_pred[j] + np.log(likelihood_feature)
                # add prior
                temp_y_pred[j] = temp_y_pred[j] + np.log(self.classes_prior[j])
            # MAP
            for l in range(self.num_classes):
                if l == 0:
                    y_pred[index] = temp_y_pred[l]
                    class_id = l
                elif temp_y_pred[l] > y_pred[index]:
                    y_pred[index] = temp_y_pred[l]
                    class_id = l
            y_pred[index] = np.unique(self.Y_train)[class_id]
        return y_pred