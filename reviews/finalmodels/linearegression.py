import pandas as pd 
import numpy as np 
import math as mt

class Myknn:
    def __init__(self, trainingdata, testdata):
        self.k_similar = 5
        self.mode = 'classification'
        self.labels = trainingdata['MEDV']  
        self.trainingdata = trainingdata.drop('NObeyesdad', axis=1)  
        self.testdata = testdata.drop('NObeyesdad', axis=1)  
        

    def __setup(self):
        self.preprocessed_traindata = self.__preprocess(self.trainingdata)
        self.preprocessed_testdata = self.__preprocess(self.testdata)
 
    def __preprocess(self):
        self.label_mean = self.labels

    def __predict(self):
        predictions = []


        return predictions
    
    def trainadnuse(self):
        self.__setup()
        return self.__predict()
        



