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
from compiler.ast import flatten


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
    with open('input/DigitRecognition/train.csv','r') as fp:
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
    with open('input/DigitRecognition/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    # remove the title
    l.remove(l[0])
    data = array(l) # m * n (m 行 * n 列，这个文件中是没有 label 的)
    return normalizing(formatToInt(data))


def saveResult(result, csvName):
    '''
    Desc:
        将我们识别出来的数字存储到 csv 文件中，得到 submission
    Args:
        result -- 我们得到的识别的结果
        csvName -- 我们写入的 csv 文件名称
    Returns:
        无
    '''
    with open(csvName,'wb') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for i in result:
            tmp=[]
            index = index + 1
            tmp.append(index)
            tmp.append(int(i))
            myWriter.writerow(tmp)


def simpleSvmClassify(trainData, trainLabel, testData):
    '''
    Desc:
        调用 sklearn 的 svm.SVC() 来进行分类
    Args:
        trainData -- 训练数据的 features
        trainLabel -- 训练数据的对应的 label
        testData -- 测试数据的 features 
    Returns:
        test_y -- 测试数据对应的 labels
    '''
    svc = svm.SVC(C=10, kernel='rbf')
    # print "trainData==>", type(trainData), shape(trainData)
    # print "trainLabel==>", type(trainLabel), shape(trainLabel)
    # print "testData==>", type(testData), shape(testData)  
    svc.fit(trainData, trainLabel)
    test_y = svc.predict(testData)
    saveResult(test_y,'output/DigitRecognizer/Result_simple_svm.csv')
    return test_y
    

def simpleDRecognition():
    loadStartTime = time.time()
    trainData,trainLabel=loadTrainData()
    nanana = flatten(trainLabel.tolist())
    newTrainLabel = array(nanana)
    loadEndTime=time.time()
    testData=loadTestData()
    print "load data finish"
    print('load data time used:%f' % (loadEndTime - loadStartTime))
    t = time.time()
    result=simpleSvmClassify(trainData,newTrainLabel,testData)  
    print "finish!"
    k=time.time()
    print('classify time used:%f' % (k - t))

if __name__ == '__main__':
    simpleDRecognition()