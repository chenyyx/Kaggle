#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017-08-03 14:04:32
@author: ApacheCN_xy
Description: DigitRecognizer for sklearn_logisticRegression
Tips:this lr use time=4178s....,so many time aaaaaaaaaa!!!!! And the Kaggle Score=0.90786
github: https://github.com/chenyyx/Kaggle
"""

from numpy import *
import csv
import time
import pandas as pd


# 加载数据
def opencsv():#使用pandas打开      
    data = pd.read_csv('input/DigitRecognition/train.csv')      
    data1=pd.read_csv('input/DigitRecognition/test.csv')
    train_data = data.values[0:,1:]#读入全部训练数据
    train_label = data.values[0:,0]
    test_data=data1.values[0:,0:]#测试全部测试个数据
    return train_data,train_label,test_data


def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])
        index=0
        for i in result:
            tmp=[]
            index=index+1
            tmp.append(index)
            #tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)


from sklearn.linear_model.logistic import LogisticRegression

def lrClassify(trainData,trainLabel,testData):
    lr = LogisticRegression()
    lr.fit(trainData, ravel(trainLabel))
    testLabel = lr.predict(testData)
    saveResult(testLabel,'output/DigitRecognizer/Result_sklearn_lr.csv')
    return testLabel

def dRecognition_lr():
    loadStartTime = time.time()
    trainData,trainLabel,testData = opencsv()
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)
    loadEndTime=time.time()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    result=lrClassify(trainData,trainLabel,testData)   
    print "finish!"
    k=time.time()
    print('classify time used:%f' % (k - t))


if __name__ == '__main__':
    dRecognition_lr()