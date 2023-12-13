import pandas as pd
import numpy as np
from reviews.data import dataprovider
datapath= "RealState.csv"
dataprovider = dataprovider(datapath, 70, 30)

class MlLinearRegression():
    def __init__(self,train_data,test_data,testdataLabel):
        self.trainlabel = train_data['MEDV']
        self.train = train_data.drop(['MEDV'],axis=1)
        self.test = test_data
        self.testdataLabel = testdataLabel

    def __normalize(self, column):
        maxnum = column.max()
        minnum = column.min()
        return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0
    
    def preprocess(self):
        for column in self.train:
            self.train[column]= self.__normalize(self.train[column])
        for column in self.test:
            self.test[column] = self.__normalize(self.test[column])
    
    def mean_features(self):
        self.mean_features = self.train.mean()
        print(self.mean_features)
    
    def nominators(self):
        self.nominator = pd.DataFrame()
        for column in self.train.columns:
            self.nominator[column] = (self.train[column] - self.train[column].mean())*(self.trainlabel-self.trainlabel.mean())
        self.nominator = self.nominator.sum(axis=0)
        print(self.nominator)

    def denominators(self):
        self.denominator = pd.DataFrame()
        for column in self.train.columns:
            self.denominator[column] = (self.train[column] - self.train[column].mean())**2
        self.denominator = self.denominator.sum(axis=0)
        print(self.denominator)
    
    def bns(self):
        self.bnss=self.nominator/self.denominator  
        print(self.bnss)
    
    def b0(self):
        self.b0 = self.trainlabel.mean() - np.dot(self.bnss, self.mean_features)
        print(self.b0)
    
    def predict(self):
        self.predictions = self.b0 + np.dot(self.test, self.bnss)
        print(self.predictions)
        print(self.testdataLabel)
        return self.predictions
    def evaluate(self):
        predictions = self.predict()
        mae = np.mean(abs(predictions - self.testdataLabel))
        mse = np.mean((predictions - self.testdataLabel) ** 2)
        r_squared = 1 - (sum((self.testdataLabel - predictions) ** 2) / sum((self.testdataLabel - self.testdataLabel.mean()) ** 2))

        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r_squared}")

        return mae, mse, r_squared
        




traindata, testdata = dataprovider.providedata()
traindata = traindata[['CRIM', 'AGE','ZN', 'MEDV','NOX']].copy()
testdataLabel = testdata['MEDV'].copy()
testdata = testdata[['CRIM', 'AGE','ZN','NOX']].copy()
MlLinearRegression = MlLinearRegression(traindata,testdata,testdataLabel)
MlLinearRegression.preprocess()
MlLinearRegression.mean_features()
MlLinearRegression.nominators()
MlLinearRegression.denominators()
MlLinearRegression.bns()
MlLinearRegression.b0()
MlLinearRegression.predict()
MlLinearRegression.evaluate()