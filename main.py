# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:55:37 2021

@author: Lemus
"""
import pandas as pd

def main():
    dataFolder = './data/'
    data = loadCSV(dataFolder + 'wine.data')
    print(data.iloc[:, 0])
    

#------------------supporting functions below------------------------#

def loadCSV(dataFile):
    preData = pd.read_csv(dataFile)
    labels = preData.iloc[:, 0]
    data = preData.iloc[:,1::]
    
