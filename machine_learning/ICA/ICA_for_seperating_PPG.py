# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:56:47 2021

ICA test code for seperating PPG signals

@author: Yingjian Song
"""
import numpy as np
import matplotlib.pyplot as plt
from ICA import ICA

def downsample(a1):
    cat1 = []
    for i in range(len(a1)):
        dog = a1[i]
        snake = []
        for j in range(1000//4):
            snake.append(dog[2+j*4])
        cat1.append(snake)
    cat1 = np.array(cat1)
    return cat1

# Get data directory
X_overall_path = 'F:/machine_learning/bp/data/new_Data/X_overall.npy'

#load data
X_overall = np.load(X_overall_path)
X_overall = downsample(X_overall)

#get source signal ids
ids = np.random.randint(0, X_overall.shape[0], 4)

#plot source signals respectively
for i in range(len(ids)):
    s = X_overall[ids[i], :]
    plt.figure(num=i+1)
    plt.plot(s)
    plt.title('source signal: %s'%(i+1))
    
#plot source signals in one figure
plt.figure(num=len(ids) + 1)
for i in range(len(ids)):
    s = X_overall[ids[i], :]
    plt.plot(s)
    plt.title('source signals')

plt.legend(['s1','s2', 's3', 's4'], loc='best')

#generate transform matrix
A = np.random.randint(1,10,(len(ids),len(ids)))

#mix up source signals
S = []
for i in range(len(ids)):
    S.append(X_overall[ids[i], :])
S = np.array(S)

# generate observed signals
X = np.dot(A, S)
plt.figure(num=len(ids) + 2)
for i in range(len(ids)):
    x = X[i, :]
    plt.plot(x)
plt.title('observed signals')
plt.legend(['x1','x2', 'x3', 'x4'], loc='best')

#perform ICA
ica = ICA(4, 0.000001, 0.01)
W, normalized_X, S_r = ica.fastICA(X)

#plot recovered signals
for i in range(len(ids)):
    plt.figure(num=len(ids) +3+i)
    sr = S_r[i, :]
    plt.plot(sr)
    plt.title('recovered signal')
