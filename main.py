# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:55:37 2021

@author: Lemus
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy
from numpy import random
def main():
    
    
    data, target, feature_names, target_name, dataFile = loadCSV('./data/wine.data')
    kNN(data, target)
    

    

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
    
def kNN(data, target):
    print(data)
    print(target)
    seed = random.randint(100)
    numpy.random.seed(seed)
    numpy.random.shuffle(data)
    numpy.random.seed(seed)
    numpy.random.shuffle(target)
    print(data[0])
    print(target[0])
    
    sample_size = data.shape[0]
    training_size = int(sample_size* 0.8)
    max_neighbors = int(numpy.sqrt(sample_size) + 3)
    for x in range(1, max_neighbors):
        print("\nfor %d neighbors: " % x)
        clf = KNeighborsClassifier(n_neighbors=x)
        clf.fit(data[:training_size], target[:training_size])
        predict = clf.predict(data[training_size:])
        print(predict)
        print(target[training_size:])
        score = clf.score(data[training_size:], target[training_size:]) 
        print(score)
    
main()