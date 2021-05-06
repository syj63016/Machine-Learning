# -*- coding: utf-8 -*-
"""
Created on Wed May  5 00:18:08 2021

This code is for testing SVM algorithm

@author: Yingjian
"""
import numpy as np
from SVM import SVM

def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        #获取当前行，并按“，”切割成字段放入列表中
        #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        #split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        #数字0标记为1  其余标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    #返回数据集和标记
    return dataArr, labelArr

def test(testDataList, testLabelList):
        '''
        测试
        :param testDataList:测试数据集
        :param testLabelList: 测试标签集
        :return: 正确率
        '''
        #错误计数值
        errorCnt = 0
        #遍历测试集所有样本
        for i in range(len(testDataList)):
            #打印目前进度
            print('test:%d:%d'%(i, len(testDataList)))
            #获取预测结果
            result = SVM.predict(testDataList[i])
            #如果预测与标签不一致，错误计数值加一
            if result != testLabelList[i]:
                errorCnt += 1
        #返回正确率
        return 1 - errorCnt / len(testDataList)

# 获取训练集及标签
print('start read transSet')
trainDataList, trainLabelList = loadData('D:/machine learning algorithm/Mnist_data/mnist_train.csv')
trainDataList = np.array(trainDataList)
trainLabelList = np.array(trainLabelList)

# 获取测试集及标签
print('start read testSet')
testDataList, testLabelList = loadData('D:/machine learning algorithm/Mnist_data/mnist_test.csv')
testDataList = np.array(testDataList)
testLabelList = np.array(testLabelList)

#初始化SVM类
print('start init SVM')
svm = SVM(100, 200, trainDataList[:1000])

k = svm.calcKernel(trainDataList[:1000], 10)
k = np.array(k)
# 开始训练
print('start to train')
W, b= svm.fit(trainDataList[:1000], trainLabelList[:1000])

print('start to predict')
predictions = svm.predict(testDataList[:1000])
# 开始测试
print('start to test')
cnt = 0
for i in range(len(testLabelList[:1000])):
    if testLabelList[:1000][i] == predictions[i]:
        cnt = cnt + 1
accuracy = cnt / len(testLabelList[:1000]) * 100
print('the accuracy is:', accuracy)