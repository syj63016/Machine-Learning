# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:58:17 2021

test code for ICA

@author: Yingjian Song
"""
from numpy import *
import math
import random
import matplotlib.pyplot as plt
from ICA import ICA

def f1(x, period = 4):
    return 0.5*(x-math.floor(x/period)*period)

def create_data():
    #data number
    n = 500
    #data time
    T = [0.1*xi for xi in range(0, n)]
    #source
    S = array([[sin(xi)  for xi in T], [f1(xi) for xi in T]], float32)
    #mix matrix
    A = array([[0.8, 0.2], [-0.3, -0.7]], float32)
    return T, S, dot(A, S)

def show_data(T, S):
    plt.plot(T, [S[0,i] for i in range(S.shape[1])], marker="*")
    plt.plot(T, [S[1,i] for i in range(S.shape[1])], marker="o")
    plt.show()

# creat signals
T, S, X = create_data()

# plot 2 signals before mixing up
fig1 = plt.figure(num = 1)
show_data(T, S)
plt.title('source signals')

# plot 2 signals after mixing up
fig2 = plt.figure(num = 2)
show_data(T, X)
plt.title('observed signals')

#perform ICA
ica = ICA(2, 0.0001, 0.01)
W, normalized_X, S_r = ica.fastICA(X)

#plot recovered signals
fig3 = plt.figure(num = 3)
show_data(T, S_r)
plt.title('recovered signals')
