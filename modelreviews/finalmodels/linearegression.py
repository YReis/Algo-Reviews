import pandas as pd
import numpy as np

class MlLinearRegression():
    def __init__(self,train_data,test_data):
        self.trainlabel = train_data['MEDV']
        self.train = train_data[['CRIM', 'AGE','ZN','NOX','MEDV']]
        self.test = test_data[['CRIM', 'AGE','ZN','NOX']]
        self.testdataLabel = test_data['MEDV']

    def __normalize(self, column):
        maxnum = column.max()
        minnum = column.min()
        return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0
    
    def __preprocess(self):
        self.train = self.train[['CRIM', 'AGE','ZN','NOX']].copy()
        for column in self.train:
            self.train[column]= self.__normalize(self.train[column])
        for column in self.test:
            self.test[column] = self.__normalize(self.test[column])
    
    def __mean_features(self):
        self.mean_features = self.train.mean()
        print(self.mean_features)
    
    def __nominators_denominators(self):
        self.nominator = pd.DataFrame()
        for column in self.train.columns:
            self.nominator[column] = (self.train[column] - self.train[column].mean())*(self.trainlabel-self.trainlabel.mean())
        self.nominator = self.nominator.sum(axis=0)
        print(self.nominator)

        self.denominator = pd.DataFrame()
        for column in self.train.columns:
            self.denominator[column] = (self.train[column] - self.train[column].mean())**2
        self.denominator = self.denominator.sum(axis=0)
        print(self.denominator)
    
    def __bns_b0(self):
        self.bnss=self.nominator/self.denominator  
        self.b0 = self.trainlabel.mean() - np.dot(self.bnss, self.mean_features)
    
    def __predict(self):
        self.predictions = self.b0 + np.dot(self.test, self.bnss)
        print(self.predictions)
        print(self.testdataLabel)
        return self.predictions
    def __evaluate(self):
        predictions = self.__predict()
        mae = np.mean(abs(predictions - self.testdataLabel))
        mse = np.mean((predictions - self.testdataLabel) ** 2)
        r_squared = 1 - (sum((self.testdataLabel - predictions) ** 2) / sum((self.testdataLabel - self.testdataLabel.mean()) ** 2))

        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r_squared}")

        return mae, mse, r_squared
    def trainanduse(self):
        self.__preprocess()
        self.__mean_features()
        self.__nominators_denominators()
        self.__bns_b0()
        self.__evaluate()

