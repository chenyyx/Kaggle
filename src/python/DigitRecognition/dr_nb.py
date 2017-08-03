#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017-8-3 18:32:16
@author: ApacheCN_xy
Description: DigitRecognizer for sklearn_naiveBayes
Tips:
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

# 调用 sklearn 的朴素贝叶斯算法包,这里只调用 GaussianNB 和 MultinomialNB
# from sklearn.naive_bayes import GaussianNB # 高斯分布的NB

# def GaussianNBClassify(trainData,trainLabel,testData): 
#     nbClf=GaussianNB()          
#     nbClf.fit(trainData,ravel(trainLabel))
#     testLabel=nbClf.predict(testData)
#     saveResult(testLabel,'output/DigitRecognizer/sklearn_GaussianNB_Result.csv')
#     return testLabel

from sklearn.naive_bayes import MultinomialNB   #nb for 多项式分布的数据    
def MultinomialNBClassify(trainData,trainLabel,testData): 
    nbClf=MultinomialNB(alpha=0.1)      #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.       
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'output/DigitRecognizer/sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return testLabel


def dRecognition_nb():
    loadStartTime = time.time()
    trainData,trainLabel,testData = opencsv()
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)
    loadEndTime=time.time()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    result=MultinomialNBClassify(trainData,trainLabel,testData)   
    print "finish!"
    k=time.time()
    print('classify time used:%f' % (k - t))


if __name__ == '__main__':
    dRecognition_nb()