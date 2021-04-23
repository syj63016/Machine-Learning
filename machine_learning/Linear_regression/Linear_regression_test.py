# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:53:10 2021

@author: Administrator
"""
#import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from Linear_regression import Linear_regression

def mean_squared_error(y_true, y_pred):
#真实数据与预测数据之间的差值（平方平均）
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

#第一步：导入数据
# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
#X = diabetes.data[:, np.newaxis, 2]
X = diabetes.data
print (X.shape)

#第二步：将数据分为训练集以及测试集
# Split the data into training/testing sets
x_train, x_test = X[:-20], X[-20:]

# Split the targets into training/testing sets
y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]

#第三步：导入线性回归类（之前定义的）
clf = Linear_regression(0.0001, 8000, regularization = 'l2', 
                        regularization_lr = 0.000001)
clf.fit(x_train, y_train)#训练
y_pred = clf.predict(x_test)#测试

#第四步：测试误差计算（需要引入一个函数）
# Print the mean squared error
print ("Mean Squared Error:", mean_squared_error(y_test, y_pred))
#matplotlib可视化输出
# Plot the results
#plt.scatter(x_test[:,0], y_test,  color='black')#散点输出
#plt.plot(x_test[:,0], y_pred, color='blue', linewidth=3)#预测输出
#plt.show()