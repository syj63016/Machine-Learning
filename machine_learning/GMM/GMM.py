1# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:42:28 2021

GMM+EM

ps:
    initialize means of data using kmeans
    initialize covariance of data as identity matrix for all clusters
    initialize weights of each clusters as 1/N, N represents num of clusters

@author: Yingjian Song
"""
import numpy as np
from sklearn.cluster import KMeans

def Multiviriate_Gussian(x, mean_x, cov_x):
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
    prob = exp_part/(pow(2 * np.pi, (len(x)/2)) * pow(np.linalg.det(cov_x), 1/2))
    return prob

class GMM():
    def __init__ (self, n_distributions, n_features, X):
        
        self.n_distributions = n_distributions
        self.z_posterior = np.zeros((X.shape[0], self.n_distributions))
        kmeans = KMeans(n_clusters=4).fit(X)
        means_x = kmeans.cluster_centers_
        
        for i in range(n_distributions):
            if i == 0:
                self.mean_overall = means_x[i]
                self.cov_overall = np.identity(n_features)
                self.cov_overall = np.expand_dims(self.cov_overall, 
                                                  axis = 0)
            else:
                X_data_mean = means_x[i]
                self.mean_overall = np.vstack((self.mean_overall,X_data_mean))
                X_data_cov = np.identity(n_features)
                X_data_cov = np.expand_dims(X_data_cov, 
                                                  axis = 0)
                self.cov_overall = np.vstack((self.cov_overall, X_data_cov))
        
        self.prior = np.ones(self.n_distributions) * (1/self.n_distributions)
    def M_step(self, X):
        self.Posterior = np.zeros(self.n_distributions)
        num_data_samples = X.shape[0]
        for k in range(self.n_distributions):
            for i in range(num_data_samples):
                self.Posterior = self.z_posterior[i,k]
                if i == 0:
                    Posterior_sum = self.Posterior
                    xPosterior_sum = self.Posterior * X[i,:]
                else:
                    Posterior_sum = self.Posterior + Posterior_sum
                    xPosterior_sum = self.Posterior * X[i,:] + xPosterior_sum
                    
            self.prior[k] =  Posterior_sum/num_data_samples            
            self.mean_overall[k, :] = xPosterior_sum/Posterior_sum
            for i in range(num_data_samples):
                self.Posterior = self.z_posterior[i,k]
                centered_x = np.reshape((X[i,:] - self.mean_overall[k,:]), (-1,1))
                if i == 0:
                    Posterior_sum = self.Posterior
                    cov_Posterior_sum = self.Posterior * np.dot(centered_x, 
                                                   centered_x.T)
                else:
                    Posterior_sum = self.Posterior + Posterior_sum
                    cov_Posterior_sum = self.Posterior * np.dot(centered_x, 
                                                   centered_x.T) + cov_Posterior_sum
            self.cov_overall[k,:,:] = cov_Posterior_sum/Posterior_sum
        return self.prior, self.mean_overall, self.cov_overall
    
    def E_step(self, X, prior, mean_overall, cov_overall):
        num_data_samples = X.shape[0]
        for i in range(num_data_samples):
                for k in range(self.n_distributions):
                    likelihood = Multiviriate_Gussian(X[i,:], mean_overall[k, :], cov_overall[k,:,:])
                    joint_prob = prior[k] * likelihood
                    for j in range(self.n_distributions):
                        if j == 0:
                            x_prob = Multiviriate_Gussian(
                                        X[i,:], 
                                        mean_overall[j],
                                        cov_overall[j]) * prior[j]
                        else:
                            x_prob = Multiviriate_Gussian(
                                        X[i,:], 
                                        mean_overall[j],
                                        cov_overall[j]) * prior[j] + x_prob
                    self.z_posterior[i,k] = joint_prob/x_prob
    
    def calculate_likelihood(self, X, prior, mean_overall, cov_overall):
        num_data_samples = X.shape[0]
        log_likelihood_e = 0
        for i in range(num_data_samples):
            for k in range(self.n_distributions):
                likelihood = Multiviriate_Gussian(X[i,:], mean_overall[k, :], cov_overall[k,:,:])
                log_joint_prob = np.log(likelihood * prior[k] + 0.0000001)
                log_likelihood_e = log_likelihood_e + log_joint_prob
        return log_likelihood_e

    def EM_training(self, X, iter_steps):
        self.log_joint_prob = 0
        last_log_joint_prob = float('-inf') 
        step = 0
        cnt = 0
        while True:
            step = step + 1
            print('>>>>>>>>>>>>>>step:', step)
            self.E_step(X, self.prior, self.mean_overall, self.cov_overall)
            self.prior, self.mean_overall, self.cov_overall = self.M_step(X)
            self.log_joint_prob = self.calculate_likelihood(X, self.prior, self.mean_overall, self.cov_overall)
            print('self.prior:')
            print(self.prior)
            print('self.mean_overall:')
            print(self.mean_overall)
            print('self.cov_overall:')
            print(self.cov_overall)
            print('last_log_joint_prob:', last_log_joint_prob)
            print('log_joint_prob:', self.log_joint_prob)
            
            if step > iter_steps:
                break
            
            if self.log_joint_prob - last_log_joint_prob < 0.0000001:
                cnt = cnt +1
            else:
                cnt = 0
            
            if cnt > 0:
                break
            
            last_log_joint_prob = self.log_joint_prob
        return self.prior, self.mean_overall, self.cov_overall