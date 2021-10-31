# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:55:37 2021

@author: Lemus
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy
from numpy import random
import matplotlib.pyplot as mpl
def main():
    
    
    data, target, feature_names, target_name, dataFile = loadCSV('./data/wine.data')
    #kNN(data, target)
    #meetTheData(data, feature_names, target_name)
    crossVal(1,data, target)
    

#------------------supporting functions below------------------------#

def loadCSV(dataFile):
    preData = pd.read_csv(dataFile)
    target = preData.iloc[:, 0]
    data = preData.iloc[:,1::]
    preNames = preData.columns
    feature_names = [x for x in preNames] #to format it properly
    target_name = feature_names[0]
    feature_names = feature_names[1::]
    
    return data.to_numpy(dtype='float'), target.__array__(), feature_names, target_name, dataFile
    

def meetTheData(data, feature_names, target_name):
    print("this method is stupid and has no use")
    print(feature_names)
    numFeatures = len(feature_names)
    numSamples = data.shape[0]
    fig, ax = mpl.subplots()
    ax.set_axis_off()
    
    mpl.show()
def kNN(data, target):
    seed = 1
    numpy.random.seed(seed)
    numpy.random.shuffle(data)
    numpy.random.seed(seed)
    numpy.random.shuffle(target)
    sample_size = data.shape[0]
    training_size = int(sample_size* 0.8)
    max_neighbors = int(numpy.sqrt(sample_size) + 3)
    testing_accuracy = numpy.zeros(max_neighbors)
    training_accuracy = numpy.zeros(max_neighbors)
    for x in range(1, max_neighbors):
        print("\nfor %d neighbors: " % x)
        clf = KNeighborsClassifier(n_neighbors=x)
        clf.fit(data[:training_size], target[:training_size])
        predict = clf.predict(data[training_size:])
        print(predict)
        print(target[training_size:])
        score = clf.score(data[training_size:], target[training_size:]) 
        testing_accuracy[x] = score
        #predict = clf.predict(data[:training_size])
        score = clf.score(data[:training_size], target[:training_size])
        training_accuracy[x] = score
    print(training_accuracy[1:])
    print(testing_accuracy)
    print(seed)
    x1 = numpy.linspace(1,max_neighbors, max_neighbors-1)
    print(x1)
    fig, ax = mpl.subplots()
    ax.plot(x1, training_accuracy[1:], label="training")
    ax.plot(x1, testing_accuracy[1:], label="testing")
    ax.legend()
    ax.set_xlabel("Number of Neighbors")
    ax.set_ylabel("Accuracy")
    
def crossVal(num_neighbors, data, target):
    skfold = StratifiedKFold(n_splits=5)
    
    logreg = LogisticRegression()
    print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, data, target, cv=skfold)))
    
    
main()