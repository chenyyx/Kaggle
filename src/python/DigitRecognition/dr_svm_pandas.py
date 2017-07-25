#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017-7-25 14:04:32
@author: ApacheCN_xy
Description: DigitRecognizer for sklearn_knn
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

from sklearn.svm import SVC
from sklearn.decomposition import PCA


def svmClassify(trainData,trainLabel,testData):
    pca = PCA(n_components=0.8, whiten=True)
    train_x = pca.fit_transform(trainData)
    test_x = pca.transform(testData)
    svc = SVC(kernel='rbf', C=10)
    svc.fit(train_x, trainLabel)
    h=time.time()
    test_y = svc.predict(test_x)
    saveResult(test_y,'output/DigitRecognizer/Result_svm.csv')
    return test_y


def dRecognition_svm():
    loadStartTime = time.time()
    trainData,trainLabel,testData = opencsv()
    loadEndTime=time.time()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    result=svmClassify(trainData,trainLabel,testData)   
    print "finish!"
    k=time.time()
    print('classify time used:%f' % (k - t))


if __name__ == '__main__':
    # dRecognition()
    dRecognition_svm()