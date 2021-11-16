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
from sklearn.model_selection import cross_validate
import numpy 
from numpy import random
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as mpl
def main():
    
    
    data, target, feature_names, target_name, dataFile = loadCSV('./data/wine.data')
    k = kNN(data, target)
    meetTheData(data, target, feature_names, target_name)
    crossVal(k,data, target)

    

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
    

def meetTheData(data, target, feature_names, target_name):
    #print("this method is stupid and has no use")
    # print(data)
    #add target name to the feature names
    feature_names.append(target_name)
    #make a vessel to hold the data with an extra column for the targets
    data_shape = [data.shape[0], data.shape[1]+1]
    dataholder = numpy.zeros(data_shape)
    dataholder[:, :-1] = data;
    dataholder[:, -1] = target


    #first five rows
    fig1, table = mpl.subplots(1 ,1)
    table.set_axis_off()
    the_table = table.table(cellText = dataholder[0:5, :],
                colLabels=feature_names,
                cellLoc = 'center',
                loc = 'center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.scale(5, 3)
    table.set_title('Examples of dataset')

    #histograms   
    alcContent = data[:, 0]
    malicContent = data[:, 1]
    fig, alcHist = mpl.subplots(1, 1)
    alcHist.hist(alcContent)
    alcHist.set_title('Alcohol Content')
    alcHist.set_xlabel("Alcohol content (%)")
    alcHist.set_ylabel("Occurences in the dataset")
    
    #scatterplot
    #get indeces of 1s, 2s, 3s in the target column 
    cultivar1indeces = numpy.where(dataholder[:, 13] == 1)
  
    cultivar2indeces = numpy.where(dataholder[:, 13] == 2)

    cultivar3indeces = numpy.where(dataholder[:, 13] == 3)


    #Cultivar 1
    cultivar1Alc = [data[i, 0] for i in cultivar1indeces]
    cultivar1Malic = [data[i, 1] for i in cultivar1indeces]
    
    #Cultivar 2
    cultivar2Alc = [data[i, 0] for i in cultivar2indeces]
    cultivar2Malic = [data[i, 1] for i in cultivar2indeces]
    
    #Cultivar 3
    cultivar3Alc = [data[i, 0] for i in cultivar3indeces]
    cultivar3Malic = [data[i, 1] for i in cultivar3indeces]
    
    
    
    fig, scatter = mpl.subplots()
    scatter.scatter(cultivar1Alc, cultivar1Malic, label='Cultivar 1')
    scatter.scatter(cultivar2Alc, cultivar2Malic, label='Cultivar 2')
    scatter.scatter(cultivar3Alc, cultivar3Malic, label='Cultivar 3')
    scatter.set_xlabel('Alcohol content (%)')
    scatter.set_ylabel('Malic acid')
    scatter.set_title('Alcohol content vs Malic acid')
    scatter.legend(fontsize = 12)

    
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
    highest_acc = 0
    highest_neigh = 0
    for x in range(1, max_neighbors):
        # print("\nfor %d neighbors: " % x)
        clf = KNeighborsClassifier(n_neighbors=x)
        clf.fit(data[:training_size], target[:training_size])
        predict = clf.predict(data[training_size:])
        # print(predict)
        # print(target[training_size:])
        score = clf.score(data[training_size:], target[training_size:]) 
        testing_accuracy[x] = score
        if(score > highest_acc):
            hightest_acc = score
            highest_neigh = x
        #predict = clf.predict(data[:training_size])
        score = clf.score(data[:training_size], target[:training_size])
        training_accuracy[x] = score
    # print(training_accuracy[1:])
    # print(testing_accuracy)
    #print(seed)
    x1 = numpy.linspace(1,max_neighbors, max_neighbors-1)
    # print(x1)
    fig, ax = mpl.subplots()
    ax.plot(x1, training_accuracy[1:], label="training")
    ax.plot(x1, testing_accuracy[1:], label="testing")
    ax.legend()
    ax.set_xlabel("Number of Neighbors")
    ax.set_ylabel("Accuracy")
    return highest_neigh
    
def crossVal(num_neighbors, data, target):
    skfold = StratifiedKFold(n_splits=5)
    reg = KNeighborsClassifier(n_neighbors=num_neighbors)

    temp = cross_validate(reg, data, target, cv=skfold,return_train_score=True)
    
    # print(temp['test_score'])
    test_score = [x for x in temp['test_score']]
    train_score = [x for x in temp['train_score']]
    
    train_and_test = numpy.zeros([2, 5])
    train_and_test[0, :] = test_score
    train_and_test[1, :] = train_score
    
    
    fig, table = mpl.subplots()
    table.set_axis_off()
    the_table = table.table(cellText = train_and_test,
                            rowLabels = ['test score', 'train score'],
                            colLabels = ['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5', ],
                            cellLoc = 'center',
                            loc = 'center'
                            )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(22)
    the_table.scale(5, 3)
    table.set_title('Fold accuracy')
    
    
    
    
main()