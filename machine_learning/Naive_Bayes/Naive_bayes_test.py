# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:33:50 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Naive_bayes import Naive_Bayes
data = load_iris()
iris_target = data.target #得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式
x_train, x_test, y_train, y_test = train_test_split(iris_features, 
                                                    iris_target, 
                                                    test_size = 0.2, 
                                                    random_state = 1234)

#set nunmbers of iterations and learning rate
classifier = Naive_Bayes()
#training
x_train = np.array(x_train)
likehoods_for_all_features = classifier.fit(x_train, y_train, [0, 0, 0, 0])
#prediction
predictions = classifier.predict(np.array(x_test))

# y_test = np.reshape(y_test, (-1,1))

accuracy = len(predictions[predictions==y_test])/len(y_test) * 100
print('classifier acurracy is:', accuracy)