#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import csv

# array format to int
def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newmat=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newmat[i,j]=int(array[i,j])
    return newmat

def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                if array[i,j] > 128:
                    array[i,j] = 2
                else:
                    array[i,j] = 1
    return array

def loadTrainData():
    l=[]
    with open('input/DigitRecognition/train.csv','r') as fp:
         lines=csv.reader(fp)
         for line in lines:
             l.append(line) #42001*785
    #remove title
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label) #label 1*42000  data 42000*784

def loadTestData():
    l=[]
    with open('input/DigitRecognition/test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))  #  data 28000*784

def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])
        index=0;
        for i in result:
            tmp=[]
            index=index+1
            tmp.append(index)
            #tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)

from sklearn.ensemble import RandomForestClassifier

def RFClassify(trainData,trainLabel,testData):
    nbCF=RandomForestClassifier(n_estimators=200,warm_start = True)
    nbCF.fit(trainData,ravel(trainLabel))
    testLabel=nbCF.predict(testData)
    saveResult(testLabel,'output/DigitRecognizer/Result.csv')
    return testLabel


def dRecognition():
    trainData,trainLabel=loadTrainData()
    print "load train data finish"
    testData=loadTestData()
    print "load test data finish"
    result=RFClassify(trainData,trainLabel,testData)   
    print "finish!"

if __name__ == '__main__':
    dRecognition()