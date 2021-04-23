# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:45:18 2021

This file is aim to test Gussian Discriminant Analysis algorithm

@author: Yingjian Song
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from GDA import Gussian_discriminant_analysis as GDA
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
classifier = GDA()
#training
x_train = np.array(x_train)
classifier.fit(x_train, y_train)
#prediction
predictions = classifier.predict(np.array(x_test))

y_test = np.reshape(y_test, (-1,1))

accuracy = len(predictions[predictions==y_test])/len(y_test) * 100
print('classifier acurracy is:', accuracy)
