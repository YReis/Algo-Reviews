import pandas as pd
import numpy as np
from reviews.data import dataprovider
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

    def nominators(self):
        self.nominator = pd.DataFrame()
        for i in range(len(self.train.columns)):
            self.nominator['med feature'+str(i)] = (self.train.iloc[:,i] - self.train.iloc[:,i].mean())*(self.trainlabel-self.trainlabel.mean())
        self.nominator = self.nominator.sum(axis=1)

    def denominators(self):
        self.denominator = pd.DataFrame()
        for i in range(len(self.train.columns)):
            print(self.train.iloc[:,i])
            self.denominator['feature'+str(i)] = (self.train.iloc[:,i] - self.train.iloc[:,i].mean())**2
        self.denominator = self.denominator.sum(axis=1)
        print(self.denominator)
    def bns(self):
        print(self.nominator,self.denominator)
        self.bnss=self.nominator/self.denominator  
        print(self.bnss)  
    def b0(self):
        j = pd.DataFrame()
        yMean = self.trainlabel.mean()
        for i in range(len(self.bnss)):
            j[i] = self.bnss.iloc[i]*self.trainlabel.iloc[i]
        j = j.sum(axis=1)
            
    def medvalues(self):
        self.medY = self.trainlabel.mean()
        self.medX = self.train.sum(axis=1)/len(self.train)
    def predict(self):
        pass
 
traindata, testdata = dataprovider.providedata()
traindata = traindata[['CRIM', 'AGE', 'MEDV']].copy()
MlLinearRegression = MlLinearRegression(traindata,testdata)
MlLinearRegression.preprocess()
MlLinearRegression.medvalues()
MlLinearRegression.nominators()
MlLinearRegression.denominators()
MlLinearRegression.bns()
MlLinearRegression.b0()
MlLinearRegression.predict()
      
