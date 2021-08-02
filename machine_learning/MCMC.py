# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:57:45 2021

MCMC by M-H sampling (assumed continuous distribution)

all distributions assumed Gussian with different mean and standard deviations

@author: ysong
"""

import numpy as np
import matplotlib.pyplot as plt

mod1=lambda t:np.random.normal(10,3,t)
# set num of iterations
N = 10000
#Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(100000)
#Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 100000, 100)]
# initialize expectation and standard deviation of pdf
mean = np.mean(observation)
std = np.std(observation)

# def proposal distribution
def proposal_dist(mean, std, x):
    exp_part = np.exp(-0.5 * np.power((x-mean) * std, 2))
    probability = exp_part/(np.power(2 * np.pi, 0.5) * std)
    return probability

# def transition model
def transition_dist(x, std, x_next):
    exp_part = np.exp(-0.5 * np.power((x_next-x) * std, 2))
    probability = exp_part/(np.power(2 * np.pi, 0.5) * std)
    return probability

# def acceptance rate for M-H sampling
def acceptance_rate_func(x, x_star, p_mean, p_std, Q_mean, Q_std):
    acceptance_rate = (proposal_dist(p_mean, p_std, x_star) * transition_dist(x_star, std, x))/(proposal_dist(p_mean, p_std, x) * transition_dist(x, std, x_star))
    if acceptance_rate > 1:
        acceptance_rate = 1
    return acceptance_rate

#M-H sampling
sampled_data = []
for i in range(len(observation)):
    sampled_data.append(observation[i])
for i in range(N):
    u = np.random.uniform(0,1,1)
    if i == 0:
        x = np.random.normal(mean,std,1)
        sampled_data.append(x)
        p_x_pre = proposal_dist(mean, std, x)
        cnt = 0
    x_star = np.random.normal(x,std,1)
    acceptance_rate = acceptance_rate_func(x, x_star, mean, std, x, std)
    # print('acceptance rate:', acceptance_rate)
    if u < acceptance_rate:
        x = x_star
        sampled_data.append(x)
        mean = np.mean(np.array(sampled_data))
        std = np.std(np.array(sampled_data))
        p_x = proposal_dist(mean, std, x)
        print(i, 'iterations')
        print('standard deviation:', std)
        if np.abs(p_x - p_x_pre) < 0.1:
            cnt = cnt + 1
        else:
            cnt = 0
        p_x_pre = p_x
    if cnt > 10:
        break


samples = np.array(sampled_data)
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(1,1,1)
ax.hist( samples,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 2: Distribution of target")

fig3 = plt.figure(figsize=(10,10))
ax = fig3.add_subplot(1,1,1)
ax.hist( samples,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 3: MCMC sampled distribution")
    