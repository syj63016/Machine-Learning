# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:51:38 2021

This is the code for tesing PCA(max in projection variance)

@author: Yingjian Song
"""

from Principle_component_analysis import Principle_Component_Analysis as PCA
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.feature_names
labels = iris.target_names

# find new feature space for data
clf = PCA()
clf.fit(X)

# transform data to new feature space, only keep 2 features in this case
X_trans = clf.transform(X,2)
np.linalg.norm(X_trans[:,1])
#plot my PCA transformed data in 2 dimension
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(X_trans[y == 0, 0], X_trans[y == 0, 1], c='r')
plt.scatter(X_trans[y == 1, 0], X_trans[y == 1, 1], c='g')
plt.scatter(X_trans[y == 2, 0], X_trans[y == 2, 1], c='b')

plt.title('my PCA')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(labels, loc='best', fancybox=True)

#compare with sklearn PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sk_PCA = pca.fit_transform(X)

#plot sklearn transformed data in 2 dimension
plt.subplot(122)
plt.scatter(sk_PCA[y == 0, 0], sk_PCA[y == 0, 1], c='r')
plt.scatter(sk_PCA[y == 1, 0], sk_PCA[y == 1, 1], c='g')
plt.scatter(sk_PCA[y == 2, 0], sk_PCA[y == 2, 1], c='b')

plt.title('sklearn PCA')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(labels, loc='best', fancybox=True)