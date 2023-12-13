import pandas as pd
import numpy as np
from reviews.data import dataprovider

datapath= "RealState.csv"
dataprovider = dataprovider(datapath, 70, 30)

class MlLinearRegression():
    def __init__(self, train_data, test_data):
        self.trainlabel = train_data['MEDV']
        self.train = train_data.drop(['MEDV'], axis=1)
        self.test = test_data
        self.coefficients = []
        self.predictions = None

    def __normalize(self, column):
        maxnum = column.max()
        minnum = column.min()
        return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0

    def preprocess(self):
        for column in self.train:
            self.train[column] = self.__normalize(self.train[column])

    def calculate_coefficients(self):
        num_rows = len(self.train)
        num_cols = len(self.train.columns)
        coefficients = [0 for _ in range(num_cols)]
        x_mean = self.train.mean()
        y_mean = self.trainlabel.mean()

        for i in range(num_cols):
            numerator = 0
            denominator = 0
            for j in range(num_rows):
                x_diff = self.train.iloc[j, i] - x_mean[i]
                y_diff = self.trainlabel.iloc[j] - y_mean
                numerator += x_diff * y_diff
                denominator += x_diff ** 2
            coefficients[i] = numerator / denominator if denominator != 0 else 0

        self.coefficients = coefficients

    def calculate_intercept(self):
        x_mean = self.train.mean()
        self.b0 = self.trainlabel.mean() - np.dot(self.coefficients, x_mean)

    def predict(self):
        self.predictions = self.b0 + np.dot(self.test, self.coefficients)
        return self.predictions

# Helper functions for evaluation metrics
def mean_squared_error(actual, predicted):
    return ((actual - predicted) ** 2).mean()

def root_mean_squared_error(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def mean_absolute_error(actual, predicted):
    return abs(actual - predicted).mean()

def r_squared(actual, predicted):
    total_error = ((actual - actual.mean()) ** 2).sum()
    residual_error = ((actual - predicted) ** 2).sum()
    return 1 - (residual_error / total_error)

# Data preparation and model training
traindata, testdata = dataprovider.providedata()
traindata = traindata[['CRIM', 'MEDV']].copy()
testdataLabel = testdata['MEDV']
testdata = testdata[['CRIM']].copy()
def normalize(column):
    maxnum = column.max()
    minnum = column.min()
    return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0

for column in testdata:
    testdata[column] = normalize(testdata[column])

ml_model = MlLinearRegression(traindata, testdata)
ml_model.preprocess()
ml_model.calculate_coefficients()
ml_model.calculate_intercept()
predictions = ml_model.predict()

# Evaluation
mse = mean_squared_error(testdataLabel, predictions)
rmse = root_mean_squared_error(testdataLabel, predictions)
mae = mean_absolute_error(testdataLabel, predictions)
r2 = r_squared(testdataLabel, predictions)

# Displaying metrics
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}")

