import pandas as pd 
import numpy as np 
import math as mt

class Myknn:
    def __init__(self, trainingdata, testdata):
        self.k_similar = 5
        self.mode = 'classification'
        self.trainingdata = trainingdata
        self.labels = trainingdata['NObeyesdad']  # Store the labels
        self.trainingdata = trainingdata.drop('NObeyesdad', axis=1)  # Drop labels from training data
        self.testdata = testdata.drop('NObeyesdad', axis=1)  
        

    def __setup(self):
        self.preprocessed_traindata = self.__preprocess(self.trainingdata)
        self.preprocessed_testdata = self.__preprocess(self.testdata)
 
    def __preprocess(self,data):
        preprocesseddataframe = data.copy()

        binary_columns = ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']
        binary_mapping = {'yes': 1, 'no': 0}
        for column in binary_columns:
            preprocesseddataframe[column] = preprocesseddataframe[column].map(binary_mapping)

        gender_mapping = {"Male": 0, "Female": 1}
        preprocesseddataframe['Gender'] = preprocesseddataframe['Gender'].map(gender_mapping)

        non_binary_categorical_variables_columns = ['CAEC', 'MTRANS', 'CALC']
        for column in non_binary_categorical_variables_columns:
            dummies = pd.get_dummies(preprocesseddataframe[column], prefix=column)
            preprocesseddataframe.drop(column, axis=1, inplace=True)
            preprocesseddataframe = pd.concat([preprocesseddataframe, dummies], axis=1)

        numerical_columns_to_be_normalized = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        for column in numerical_columns_to_be_normalized:
            preprocesseddataframe[column] = self.__normalize_column(preprocesseddataframe[column])
        return preprocesseddataframe

    def __normalize_column(self, col):
        return (col - col.min()) / (col.max() - col.min())
    
    def __euclidiandistance(self,data,query):
        return mt.sqrt(((data - query)**2).sum())

    def __predict(self):
        predictions = []

        for _, test_instance in self.preprocessed_testdata.iterrows():
            distances = self.preprocessed_traindata.apply(lambda row: self.__euclidiandistance(row, test_instance), axis=1)
            nearest_neighbors = distances.nsmallest(self.k_similar)

            if self.mode == 'classification':
                # Use stored labels for determining the most common class
                most_common = self.labels.loc[nearest_neighbors.index].mode()[0]
                predictions.append(most_common)
            else:
                # For regression (not applicable in this scenario)
                average = self.labels.loc[nearest_neighbors.index].mean()
                predictions.append(average)

        return predictions
    
    def trainadnuse(self):
        self.__setup()
        return self.__predict()
        



