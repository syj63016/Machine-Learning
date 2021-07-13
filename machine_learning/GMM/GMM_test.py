# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:48:37 2021

@author: Administrator
"""

import GMM as GMM
import numpy as np

def Multiviriate_Gussian(x, mean_x, cov_x):
    # try:
    #     np.linalg.inv(cov_x)
    # except:
    #     for i in range(cov_x.shape[0]):
    #         cov_x[i, i] = cov_x[i, i] + 0.0000001
    #     cov_x = cov_x
    # else:
    #     cov_x = cov_x
    # norm = multivariate_normal(mean=mean_x, cov=cov_x)
    # return norm.pdf(x)
    centered_data = np.reshape((x-mean_x), (-1,1))
    try: 
        temp = np.linalg.inv(cov_x)
    except:
        for i in range(cov_x.shape[0]):
            cov_x[i, i] = cov_x[i, i] + 0.0000001
        inv_cov = np.linalg.inv(cov_x)
    else:
        inv_cov = temp
    exp_part = np.exp(-1/2 * np.dot(np.dot((centered_data.T), 
                                    inv_cov),
                                    centered_data))
    prob = exp_part/(pow(2 * np.pi, (len(x)/2)) * pow(np.linalg.det(cov_x + np.eye(cov_x.shape[0]) * 0.001), 1/2))
    return prob

def creat_Data(n_features, n_samples_per_distributions):
    
    n_distributions = len(n_samples_per_distributions)
    n_data_samples_overall = np.sum(n_samples_per_distributions)
    prior = n_samples_per_distributions/n_data_samples_overall
    for i in range(n_distributions):
        if i == 0:
            overall_data = np.random.randint(0,10,
                            (n_samples_per_distributions[0], n_features))
            mean_overall = np.mean(overall_data, axis = 0)
            cov_overall = np.cov(overall_data.T)
            cov_overall = np.expand_dims(cov_overall, axis = 0)
            overall_data = np.insert(overall_data, n_features, values=i, axis=1)
        else:
            X_data = np.random.randint(0,10,(n_samples_per_distributions[i], 
                                         n_features))
            X_data_mean = np.mean(X_data, axis = 0)
            X_data_cov = np.cov(X_data.T)
            X_data_cov = np.expand_dims(X_data_cov, axis = 0)
            mean_overall = np.vstack((mean_overall, X_data_mean))
            cov_overall = np.vstack((cov_overall, X_data_cov))
            X_data = np.insert(X_data, n_features, values=i, axis=1)
            overall_data = np.vstack((overall_data, X_data))
    
    np.random.shuffle(overall_data)
    lables = overall_data[:,-1]
    overall_data = overall_data[:,0:-1]
    return overall_data, prior, mean_overall, cov_overall, lables

############################################# test 1
from sklearn.datasets import load_iris

iris = load_iris()   	# 加载数据集
features = iris.data	# 获取特征集
iris_labels = iris.target    # 获取目标集

from sklearn.mixture import GaussianMixture
labels = GaussianMixture(n_components=3, covariance_type='full').fit_predict(features) #指定聚类中心个数为4

# My GMM
data_x = features
# np.random.shuffle(data_x)
GMM_clf = GMM.GMM(3, 4, data_x)
predicted_prior, predicted_mean_overall, predicted_cov_overall = GMM_clf.EM_training(data_x,
                                                                60)
my_predicted_lables = np.zeros(features.shape[0])

for i in range(features.shape[0]):
    for k in range(3):
        if k == 0 :
            max_prob = predicted_prior[k] * Multiviriate_Gussian(features[i,:], predicted_mean_overall[k], 
                                                         predicted_cov_overall[k])
            my_predicted_lables[i] = k
        else:
            prob = predicted_prior[k] * Multiviriate_Gussian(features[i,:], predicted_mean_overall[k], 
                                                         predicted_cov_overall[k])
            if prob > max_prob:
                max_prob = prob
                my_predicted_lables[i] = k

############################################# test 2
X_data, X_prior, X_mean_overall, X_cov_overall, X_lables = creat_Data(3, [300, 300, 400])
GMM_clf_2 = GMM.GMM(3, 3, X_data)
predicted_prior, predicted_mean_overall, predicted_cov_overall = GMM_clf_2.EM_training(X_data,
                                                                100)