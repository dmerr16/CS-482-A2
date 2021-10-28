# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:55:37 2021

@author: Lemus
"""
import pandas as pd

def main():
    
    
    data, target, feature_names, target_name, dataFile = loadCSV('./data/wine.data')


    

#------------------supporting functions below------------------------#

def loadCSV(dataFile):
    preData = pd.read_csv(dataFile)
    target = preData.iloc[:, 0]
    data = preData.iloc[:,1::]
    preNames = preData.columns
    feature_names = [x for x in preNames] #to format it properly
    target_name = feature_names[0]
    feature_names = feature_names[1::]
    
    return data, target, feature_names, target_name, dataFile
    
