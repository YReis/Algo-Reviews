import pandas as pd
import math
import numpy as np
import json
raw_data=pd.read_csv('../ObesityDataSet.csv')

columns_to_be_normalized = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]


def nomalize(num, minnumero, maxnumero):
    normalizednumber = (
        (num - minnumero) / (maxnumero - minnumero) if maxnumero != minnumero else 0
    )
    return normalizednumber


def applynormalization(dataframe):

    for column in dataframe.columns:
        if column == 'CALC':
            ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
            dataframe[column] = dataframe[column].map(ordinal_map)
        elif column == 'CAEC':
            mtrans_dummies = pd.get_dummies(dataframe['CAEC'], prefix='CAEC')
            dataframe.drop('CAEC', axis=1, inplace=True)
            dataframe = pd.concat([dataframe, mtrans_dummies], axis=1)
        elif column == 'MTRANS':
            mtrans_dummies = pd.get_dummies(dataframe['MTRANS'], prefix='MTRANS')
            dataframe.drop('MTRANS', axis=1, inplace=True)
            dataframe = pd.concat([dataframe, mtrans_dummies], axis=1)
        elif column in ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']:
            dataframe[column] = dataframe[column].map({'yes': 1, 'no': 0})
        elif column == "Gender":
            dataframe[column] = dataframe[column].map({"Male": 0, "Female": 1})
        elif column in columns_to_be_normalized:
            max_num = dataframe[column].max()
            min_num = dataframe[column].min()
            dataframe[column] = dataframe[column].apply(
                lambda x: nomalize(x, min_num, max_num))
            
        for column in dataframe.columns:
            if dataframe[column].dtype == bool:
                dataframe[column] = dataframe[column].astype(int)
    return dataframe

def euclidiandistance(data, query): 
    result = np.sqrt(((data - query) ** 2).sum()) 
    return result

testData = applynormalization(raw_data)
testData = testData.drop(['NObeyesdad'],axis=1)


num_rows = len(testData)
distance_dict = {}
for i in range(num_rows):
    distance_dict[i] = {}
    for j in range(num_rows):
        if i != j:
            dist = euclidiandistance(testData.iloc[i], testData.iloc[j])
            distance_dict[i][j] = dist
print(distance_dict)

