import pandas as pd
from data.dataprovider import dataprovider
def normalize(column):
    maxnum = column.max()
    minnum = column.min()
    return (column - minnum) / (maxnum - minnum) if maxnum != minnum else 0

def impute_nan_with_median(dataframe):
    return dataframe.fillna(dataframe.median())

# Load and split the data
datapath = 'RealState.csv'
dataprovider = dataprovider(datapath, 70, 30)
traindata, testdata = dataprovider.providedata()

# Select only CRIM and AGE features along with MEDV
traindata = traindata[['CRIM', 'AGE', 'MEDV']].copy()
testdata = testdata[['CRIM', 'AGE', 'MEDV']].copy()

# Impute NaN values in both train and test data
traindata = impute_nan_with_median(traindata)
testdata = impute_nan_with_median(testdata)

# Extract label before dropping from train data
traindataLabel = traindata['MEDV']
traindata = traindata.drop(['MEDV',], axis=1)

# Normalize features in training data
for column in traindata.columns:
    traindata[column] = normalize(traindata[column])

mean_label = traindataLabel.mean()

# Initialize lists to store the calculations
coefficients = []

# Calculate coefficients for CRIM and AGE
for column in traindata.columns:
    x = traindata[column]
    mean_x = x.mean()
    x_diff = x - mean_x
    y_diff = traindataLabel - mean_label

    nominator = sum(x_diff * y_diff)
    denominator = sum(x_diff * x_diff)
    
    slope = nominator / denominator if denominator != 0 else 0
    intercept = mean_label - slope * mean_x
    coefficients.append((slope, intercept))

# Normalize test data using the same parameters as training data
for column in testdata.columns:
    if column != 'MEDV':
        testdata[column] = normalize(testdata[column])

# Make predictions on the test dataset
testdata['Predicted_MEDV'] = 0
for idx, column in enumerate(traindata.columns):
    slope, intercept = coefficients[idx]
    testdata['Predicted_MEDV'] += slope * testdata[column] + intercept

# Calculate Mean Squared Error and R-squared
actual = testdata['MEDV']
predicted = testdata['Predicted_MEDV']

mse = ((predicted - actual) ** 2).mean()
ss_res = ((predicted - actual) ** 2).sum()
ss_tot = ((actual - actual.mean()) ** 2).sum()
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

print("R-squared:", r2)
print("Mean Squared Error:", mse)
