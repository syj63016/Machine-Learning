# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:40:28 2021

ICA test code for seperating ECG signals

@author: Yingjian Song
"""
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from ICA import ICA

ECG_Data_path = 'F:/machine_learning/UCI_data'
data_file_all = os.listdir(ECG_Data_path)
#get all ECG files
ECG_datafiles_all = []
for i in range(len(data_file_all)):
    ECG_datafiles_all.append(os.path.join(ECG_Data_path, data_file_all[i]))

ECG_all = []
for i in range(len(ECG_datafiles_all)):
    ECG_data = h5py.File(ECG_datafiles_all[i])
    key = list(ECG_data.keys())[-1]
    index = np.random.randint(0, ECG_data[key].shape[0])
    ECG_sig = ECG_data[key][index][0]
    ECG_sig = ECG_data[ECG_sig][:,2]
    ECG_all.append(ECG_sig)

ECG_all = np.array(ECG_all)

for i in range(len(ECG_all)):
    length = ECG_all[i].shape[0]
    if i==0:
        cut_length = length
    elif length < cut_length:
        cut_length = length

#plot source signals respectively
for i in range(len(ECG_all)):
    s = ECG_all[i][:cut_length]
    plt.figure(num=i+1)
    plt.plot(s)
    plt.title('source signal: %s'%(i+1))
    
#plot source signals in one figure
plt.figure(num=len(ECG_all) + 1)
for i in range(len(ECG_all)):
    s = ECG_all[i][:cut_length]
    plt.plot(s)
    plt.title('source signals')

plt.legend(['s1','s2', 's3', 's4'], loc='best')

#generate transform matrix
A = np.random.randint(1,100,(len(ECG_all),len(ECG_all)))

#mix up source signals
S = []
for i in range(len(ECG_all)):
    S.append(ECG_all[i][:cut_length])
S = np.array(S)

# generate observed signals
X = np.dot(A, S)
plt.figure(num=len(ECG_all) + 2)
for i in range(len(ECG_all)):
    x = X[i, :]
    plt.plot(x)
plt.title('observed signals')
plt.legend(['x1','x2', 'x3', 'x4'], loc='best')

#perform ICA
ica = ICA(4, 0.000001, 0.0001)
W, normalized_X, S_r = ica.fastICA(X)

S_r = -S_r
S_r[3] = -S_r[3]
#plot recovered signals
for i in range(len(ECG_all)):
    plt.figure(num=len(ECG_all) +3+i)
    sr = S_r[i, :]
    plt.plot(sr)
    plt.title('recovered signal')

# plt.figure(num=8)
# plt.plot(-S_r[1,:])
# plt.title('recovered signal')