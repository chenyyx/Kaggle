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
    l = []
    with open('input/DigitRecognition/train.csv','r') as fp:
        lines = csv.reader(fp)
        for line in lines:
            l.append(line)


if __name__ == '__main__':


# def loadTrainData():
#     l=[]
#     with open('input/DigitRecognition/train.csv','r') as fp:
#          lines=csv.reader(fp)
#          for line in lines:
#              l.append(line) #42001*785
#     #remove title
#     l.remove(l[0])
#     l=array(l)
#     label=l[:,0]
#     data=l[:,1:]
#     return nomalizing(toInt(data)),toInt(label) #label 1*42000  data 42000*784

# def loadTestData():
#     l=[]
#     with open('input/DigitRecognition/test.csv') as file:
#          lines=csv.reader(file)
#          for line in lines:
#              l.append(line)#28001*784
#     l.remove(l[0])
#     data=array(l)
#     return nomalizing(toInt(data))  #  data 28000*784

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