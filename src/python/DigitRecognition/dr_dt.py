#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017-8-4 17:29:57
@author: ApacheCN_xy
Description: DigitRecognizer for sklearn_decisionTree
Tips: LoadData time use 8.984000s,and classify time use 15.467000s, the Kaggle Score=0.85729
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



from sklearn.tree import DecisionTreeClassifier
def dtClassify(trainData,trainLabel,testData):
    dtClf=DecisionTreeClassifier()
    dtClf.fit(trainData,ravel(trainLabel))
    testLabel=dtClf.predict(testData)
    saveResult(testLabel,'output/DigitRecognizer/Result_sklearn_dt.csv')
    return testLabel


def dRecognition_dt():
    loadStartTime = time.time()
    trainData,trainLabel,testData = opencsv()
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)
    loadEndTime=time.time()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    result=dtClassify(trainData,trainLabel,testData)   
    print "finish!"
    k=time.time()
    print('classify time used:%f' % (k - t))


if __name__ == '__main__':
    dRecognition_dt()