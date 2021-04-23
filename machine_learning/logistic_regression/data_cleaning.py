# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 08:56:09 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
# import seaborn as sns

train_data = pd.read_csv('F:\\machine_learning\\titanic_data\\train.csv')
train_data.head()
train_data.info()
train_data.isnull().sum()

test_data = pd.read_csv('F:\\machine_learning\\titanic_data\\test.csv')
test_data.head()
test_data.info()
test_data.isnull().sum()

all_data = pd.concat([train_data, test_data], ignore_index=True)
all_data.isnull().sum()
all_data['Title'] = all_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
title_mapping = {
        'Mr': 1,
        'Miss': 2,
        'Mrs': 3,
        'Master': 4,
        'Rev': 5,
        'Dr': 6,
        'Col': 7,
        'Mlle': 8,
        'Ms': 9,
        'Major': 10,
        'Don': 11,
        'Countess': 12,
        'Mme': 13,
        'Jonkheer ': 14,
        'Sir': 15,
        'Dona': 16,
        'Capt': 17,
        'Lady': 18,
    }
titles = all_data['Name'].apply(get_title)
for k, v in title_mapping.items():
    titles[titles == k] = v
        # print(k, v)
all_data['Title'] = titles
