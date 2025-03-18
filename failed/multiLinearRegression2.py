import pandas as pd
import numpy as np
from modelreviews.data import dataprovider
datapath= "RealState.csv"
dataprovider = dataprovider(datapath, 70, 30)

class MlLinearRegression():
    def __init__(self,train_data,test_data):
     self.trainlabel = train_data['MEDV']
     self.train = train_data.drop(['MEDV'],axis=1)
     self.test = test_data

    def __normalize(self, column):
        maxnum = column.max()
        minnum = column.min()
        return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0
    
    def preprocess(self):
        for column in self.train:
            self.train[column]= self.__normalize(self.train[column])
            
    def medvalues(self):
        pass

    def predict(self):
        pass
 
traindata, testdata = dataprovider.providedata()
traindata = traindata[['CRIM', 'AGE', 'MEDV']].copy()
MlLinearRegression = MlLinearRegression(traindata,testdata)
MlLinearRegression.preprocess()
MlLinearRegression.medvalues()
MlLinearRegression.predict()
      
