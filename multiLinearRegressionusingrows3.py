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
    
    def meanrows(self):
        row_means = []
        for _, row in self.train.iterrows():
            row_means.append(row.mean())
        self.row_means =  row_means
        # print(self.row_means)
    
    def nominators(self):
        row_nominators = []
        for _,row in self.train.iterrows():
            row_x = []
            for i in range(len(row)):
                row_x.append((row[i]-self.row_means[i])*(self.trainlabel[i]-self.trainlabel.mean()))
            row_nominators.append(row_x)
        nominatorsfinal = []
        for row in row_nominators:
            nominatorsfinal.append(sum(row))
        self.eachnominator = nominatorsfinal
        print(self.eachnominator)

    def b0(self):
        self.allbnss = []
        for i in range(len(self.eachnominator)):
            self.allbnss.append(self.eachnominator[i]/self.eachdenominator[i])

        eachProduct = []
        for i in range(len(self.allbnss)):
            eachProduct.append(-(self.allbnss[i]*self.row_means[i]))
        sumationOfEachProduct = sum(eachProduct)
        self.b0final = self.trainlabel.mean() + sumationOfEachProduct
        print(self.b0final)

    def denominators(self):
        row_denominators = []
        for _,row in self.train.iterrows():
            row_x = []
            for i in range(len(row)):
                row_x.append((row[i]-self.row_means[i])**2)
            row_denominators.append(row_x)
        denominatorsfinal = []
        for row in row_denominators:
            denominatorsfinal.append(sum(row))
        self.eachdenominator = denominatorsfinal
        print(self.eachdenominator)

    def bns(self):
        self.bnss = []
        for i in range(len(self.eachnominator)):
            self.bnss.append(self.eachnominator[i]/self.eachdenominator[i])
        print(self.bnss)

    def predict(self):
        predictions = []
        for _, row in self.test.iterrows():
            y_pred = self.b0final
            for i in range(len(row)):
                y_pred += self.bnss[i] * row[i]
            predictions.append(y_pred)
        return predictions
    def calculate_metrics(self,y_true, y_pred):
        n = len(y_true)
        mse = sum((y_true - y_pred) ** 2) / n
        mae = sum(abs(y_true - y_pred)) / n
        ssr = sum((y_pred - y_true.mean()) ** 2)
        sst = sum((y_true - y_true.mean()) ** 2)
        r2 = ssr / sst
        return mse, mae, r2
    
traindata, testdata = dataprovider.providedata()
traindata = traindata[['CRIM', 'AGE', 'MEDV']].copy()
testdataLabel = testdata['MEDV'].copy()
testdata = testdata[['CRIM', 'AGE']].copy()
MlLinearRegression = MlLinearRegression(traindata,testdata,testdataLabel)
MlLinearRegression.preprocess()
MlLinearRegression.meanrows()
MlLinearRegression.nominators()
MlLinearRegression.denominators()
MlLinearRegression.bns()
MlLinearRegression.b0()
predictvalues=MlLinearRegression.predict()
mse, mae, r2 = MlLinearRegression.calculate_metrics(testdataLabel, predictvalues)
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)


      
