#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 2017-7-25 17:39:01
@author:ApacheCN_xy
Desc: using sklearn to handle Digit Recognition
Address: http://github.com/chenyyx/Kaggle
'''

from numpy import *
import csv
import time
from sklearn import svm


def formatToInt(array):
    '''
    Desc：
        将从 csv 文件中读取的 string 数据转化为 int 类型的数据
    Args:
        array -- 需要格式化的 array
    Returns:
        newMat -- 格式化为 int 类型的新的 array
    '''
    array = mat(array)
    m,n = shape(array)
    newMat = zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newMat[i, j] = int(array[i, j])
    return newMat


def normalizing(array):
    '''
    Desc:
        因为从 csv 文件中读取的数据特征太多，所以需要对数据进行归一化处理
    Args:
        array -- 需要对数据进行归一化处理的 array
    Returns:
        newArray -- 归一化处理完成后的新的 array
    '''
    m,n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                if array[i, j] > 128:
                    array[i, j] = 2
                else:
                    array[i, j] = 1
    return array


def loadTrainData():
    '''
    Desc:
        加载训练数据 -- train
    Args:
        none
    Returns:
        normalizing(formatToInt(data)) -- 返回归一化的训练数据的 features
        formatToInt(label) -- 转换为 Int 类型的 label
    '''
    l = []
    with open('input/DigitRecognition/simple_train.csv','r') as fp:
        lines = csv.reader(fp)
        for line in lines:
            l.append(line)
    # remove title since the first line is the title
    l.remove(l[0])
    l = array(l)
    # print "load train data array l is ", l, "and that's type is ==", type(l), "shape is ", shape(l)
    label = l[:,0] # label is a 1*m(行数) matrix
    # print "load train data --- label is ", label, "and that's type is ==", type(label), "shape is ", shape(label)
    data = l[:,1:]
    # print "load train data --- data is ", data, "and that's type is ==", type(data), "shape is ", shape(data)
    return normalizing(formatToInt(data)), formatToInt(label)


def loadTestData():
    '''
    Desc:
        加载测试数据 -- test
    Args:
        none
    Returns:
        normalizing(formatToInt(data)) -- 返回归一化的测试数据的 features
    '''
    l = []
    with open('input/DigitRecognition/simple_test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    # remove the title
    l.remove(l[0])
    data = array(l) # m * n (m 行 * n 列，这个文件中是没有 label 的)
    return normalizing(formatToInt(data))



if __name__ == '__main__':
    loadTrainData()



# def saveResult(result,csvName):
#     with open(csvName,'wb') as myFile:    
#         myWriter=csv.writer(myFile)
#         myWriter.writerow(["ImageId","Label"])
#         index=0;
#         for i in result:
#             tmp=[]
#             index=index+1
#             tmp.append(index)
#             #tmp.append(i)
#             tmp.append(int(i))
#             myWriter.writerow(tmp)


# def RFClassify(trainData,trainLabel,testData):
#     nbCF=RandomForestClassifier(n_estimators=200,warm_start = True)
#     nbCF.fit(trainData,ravel(trainLabel))
#     testLabel=nbCF.predict(testData)
#     saveResult(testLabel,'output/DigitRecognizer/Result.csv')
#     return testLabel


# def dRecognition():
#     trainData,trainLabel=loadTrainData()
#     print "load train data finish"
#     testData=loadTestData()
#     print "load test data finish"
#     result=RFClassify(trainData,trainLabel,testData)   
#     print "finish!"

# if __name__ == '__main__':
#     dRecognition()