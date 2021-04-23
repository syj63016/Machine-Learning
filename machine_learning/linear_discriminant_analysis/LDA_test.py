# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:22:17 2021

@author: Administrator
"""


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
# from Linear_Discriminant_Analysis import Linear_Discriminant_Analysis 

class Linear_Discriminant_Analysis(object):
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        y_classes = []
        for i in y_train:
         if i not in y_classes:
              y_classes.append(i)
        class_mean_vector = []
        # compute mean for each classes
        for i in y_classes:
            class_mean_vector.append(np.mean(X_train[y_train == i], axis = 0))
        # compute within-class matrix
        s_w = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i in range(len(y_classes)):
            temp = X_train[y_train == y_classes[i]] - class_mean_vector[i]
            n_samples_for_each_class = temp.shape[0]
            s_w = s_w + (np.dot(np.mat(temp).T, np.mat(temp))/n_samples_for_each_class)
        # compute between-class matrix
        s_b = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i in range(len(y_classes)):
            overall_mean = np.mean(X_train, axis = 0)
            temp_mean = np.mat(class_mean_vector[i] - overall_mean[i])
            s_b = s_b + np.dot(temp_mean.T, temp_mean)
        # compute eigenvalue
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(s_w) * s_b)
        # sort eigenvector based on eigenvalues
        eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        # find new feature space
        self.W = np.zeros((eigvecs.shape))
        for i in range(len(eig_pairs)):
            temp_eig = np.reshape(eig_pairs[i][1], (self.W[:,i].shape))
            self.W[:,i] = temp_eig
        
    def transform(self, X_test):
        self.X_test = X_test
        predictons = np.dot(self.X_test, self.W)
        return predictons

# prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.feature_names
labels = iris.target_names
y_c = np.unique(y)


"""visualize the distributions of the four different features in 1-dimensional histograms"""
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, column in zip(axes.ravel(), range(X.shape[1])):
    # set bin sizes
    min_b = math.floor(np.min(X[:, column]))
    max_b = math.ceil(np.max(X[:, column]))
    bins = np.linspace(min_b, max_b, 25)
 
    # plotting the histograms
    for i, color in zip(y_c, ('blue', 'red', 'green')):
        ax.hist(X[y == i, column], color=color, label='class %s' % labels[i],
                bins=bins, alpha=0.5, )
    ylims = ax.get_ylim()
 
    # plot annotation
    l = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    l.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims) + 2])
    ax.set_xlabel(names[column])
    ax.set_title('Iris histogram feature %s' % str(column + 1))
 
    # hide axis ticks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=True, labelleft=True)
 
    # remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')
fig.tight_layout()
plt.show()


np.set_printoptions(precision=4)

clf = Linear_Discriminant_Analysis()
clf.fit(X,y)
X_trans = clf.transform(X)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(X_trans[y == 0, 0], X_trans[y == 0, 1], c='r')
plt.scatter(X_trans[y == 1, 0], X_trans[y == 1, 1], c='g')
plt.scatter(X_trans[y == 2, 0], X_trans[y == 2, 1], c='b')
plt.title('my LDA')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(labels, loc='best', fancybox=True)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
X_trans2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
plt.subplot(122)
plt.scatter(X_trans2[y == 0, 0], X_trans2[y == 0, 1], c='r')
plt.scatter(X_trans2[y == 1, 0], X_trans2[y == 1, 1], c='g')
plt.scatter(X_trans2[y == 2, 0], X_trans2[y == 2, 1], c='b')
plt.title('sklearn LDA')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(labels, loc='best', fancybox=True)

# mean_vector = []  # 类别的平均值
# for i in y_c:
#     mean_vector.append(np.mean(X[y == i], axis=0))
#     print('Mean Vector class %s:%s\n' % (i, mean_vector[i]))
# """within-class scatter matrix"""
# S_W = np.zeros((X.shape[1], X.shape[1]))
# for i in y_c:
#     Xi = X[y == i] - mean_vector[i]
#     S_W += np.mat(Xi).T * np.mat(Xi)
# print('within-class scatter matrix:\n', S_W)
 
# """between-class scatter matrix """
# S_B = np.zeros((X.shape[1], X.shape[1]))
# mu = np.mean(X, axis=0)  # 所有样本平均值
# for i in y_c:
#     Ni = len(X[y == i])
#     S_B += Ni * np.mat(mean_vector[i] - mu).T * np.mat(mean_vector[i] - mu)
# print('within-class scatter matrix:\n', S_B)
# eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) * S_B)  # 求特征值，特征向量
# np.testing.assert_array_almost_equal(np.mat(np.linalg.inv(S_W) * S_B) * np.mat(eigvecs[:, 0].reshape(4, 1)),
#                                       eigvals[0] * np.mat(eigvecs[:, 0].reshape(4, 1)), decimal=6, err_msg='',
#                                       verbose=True)
# eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
# eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# W = np.zeros((eigvecs.shape))
# for i in range(len(eig_pairs)):
#     W[:,i] = eig_pairs[i][1]

