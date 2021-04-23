# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:19:28 2021

@author: Yingjian Song
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import logistic_regression
import matplotlib.pyplot as plt

data = load_iris()
iris_target = data.target #得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式

iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]


x_train, x_test, y_train, y_test = train_test_split(iris_features_part, 
                                                    iris_target_part, 
                                                    test_size = 0.2, 
                                                    random_state = 1234)

#set nunmbers of iterations and learning rate
classifier = logistic_regression(n_iters = 10000, lr = 0.0001)
#training
x_train = np.array(x_train)
normolized_X_train = (x_train-np.mean(x_train,axis = 0)) / np.std(x_train,axis = 0)
classifier.fit(x_train, y_train)
#prediction
predictions, W, loss = classifier.predict(np.array(x_test))
plt.plot(loss)

y_test = np.reshape(y_test, (-1,1))

accuracy = len(predictions[predictions==y_test])/len(y_test) * 100
print('classifier acurracy is:', accuracy)
