#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017-7-25 14:04:32
@author: ApacheCN_xy
Description: DigitRecognizer for sklearn_svm
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

# 由于 sklearn 提供了几种 svm 的 kernel ，可以试着使用其他的 kernel ，比如 ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
def svmClassify(trainData,trainLabel,testData):
    '''
    Desc:
        使用 sklearn 提供的 svm 进行分类，使用不同的 kernel 进行测试,保持其他参数不变的情况下，rbf是其中最精确的，如下：
        其中， kernel='rbf', Kaggle Score=0.98529
            kernel='linear', Kaggle Score=0.92771 ,loadData time=8.708000,classify time=321.450000
            kernel='poly', Kaggle Score=0.97757, loadData time=9.311000, classify time=123.567000
            kernel='sigmoid', Kaggle Score=0.78414, loadData time=8.875000, classify time=97.245000
            kernel='precomputed', Kaggle Score= there are something wrong ~ i will fix it later~~~
    Args:
        trainData -- <type 'numpy.ndarray'> 训练数据的 features
        trainLabel -- <type 'numpy.ndarray'> 训练数据的 Label
        testData -- <type 'numpy.ndarray'> 测试数据的 features
    Returns:
        test_y -- 测试数据完成分类的 Label
    '''
    pca = PCA(n_components=0.8, whiten=True)
    train_x = pca.fit_transform(trainData)
    test_x = pca.transform(testData)
    svc = SVC(kernel='sigmoid', C=10)
    svc.fit(train_x, trainLabel)
    h=time.time()
    test_y = svc.predict(test_x)
    saveResult(test_y,'output/DigitRecognizer/Result_SVM_Pandas_sigmoid.csv')
    return test_y


def dRecognition_svm():
    loadStartTime = time.time()
    trainData,trainLabel,testData = opencsv()
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)
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