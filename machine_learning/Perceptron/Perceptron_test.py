# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:23:22 2021

@author: Administrator
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from  Perceptron import Perceptron
#import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iris = datasets.load_iris()
x, y = iris.data, iris.target
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
    elif y[i] == 2:
        y[i] = 1
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
y_test = np.reshape(y_test, (-1,1))
clf = Perceptron(learning_rate = 0.0001, n_iters = 1000)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print("LR classification accuracy:", accuracy(y_test, predictions))
