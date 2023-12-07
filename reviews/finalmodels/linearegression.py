import pandas as pd

class MlLinearRegressor:
    def __init__(self, train_split, test_split):
        self.traindata = train_split
        self.testdata = test_split
        self.coefficients = []

    def impute_nan_with_median(self, dataframe):
        return dataframe.fillna(dataframe.median())

    def normalize(self, column):
        maxnum = column.max()
        minnum = column.min()
        return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0

    def prepare_data(self):
        # Select only CRIM and AGE features along with MEDV
        self.traindata = self.traindata[['CRIM', 'AGE', 'MEDV']].copy()
        self.testdata = self.testdata[['CRIM', 'AGE', 'MEDV']].copy()

        # Impute NaN values in both train and test data
        self.traindata = self.impute_nan_with_median(self.traindata)
        self.testdata = self.impute_nan_with_median(self.testdata)

        # Normalize features in training data
        for column in self.traindata.columns:
            if column != 'MEDV':
                self.traindata[column] = self.normalize(self.traindata[column])

    def calculate_coefficients(self):
        traindataLabel = self.traindata['MEDV']
        mean_label = traindataLabel.mean()
        self.traindata = self.traindata.drop(['MEDV'], axis=1)

        # Calculate coefficients for CRIM and AGE
        for column in self.traindata.columns:
            x = self.traindata[column]
            mean_x = x.mean()
            x_diff = x - mean_x
            y_diff = traindataLabel - mean_label

            nominator = sum(x_diff * y_diff)
            denominator = sum(x_diff * x_diff)

            slope = nominator / denominator if denominator != 0 else 0
            intercept = mean_label - slope * mean_x
            self.coefficients.append((slope, intercept))

    def predict(self):
        # Normalize test data using the same parameters as training data
        for column in self.testdata.columns:
            if column != 'MEDV':
                self.testdata[column] = self.normalize(self.testdata[column])

        # Make predictions on the test dataset
        self.testdata['Predicted_MEDV'] = 0
        for idx, column in enumerate(self.traindata.columns):
            slope, intercept = self.coefficients[idx]
            self.testdata['Predicted_MEDV'] += slope * self.testdata[column] + intercept

    def calculate_performance_metrics(self):
        # Calculate Mean Squared Error and R-squared
        actual = self.testdata['MEDV']
        predicted = self.testdata['Predicted_MEDV']

        mse = ((predicted - actual) ** 2).mean()
        ss_res = ((predicted - actual) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

        return r2, mse
    def trainadnuse(self):
        self.prepare_data()
        self.calculate_coefficients()
        self.predict()
        r2, mse = self.calculate_performance_metrics()
        return r2, mse
# Usage



