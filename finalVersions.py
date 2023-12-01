import pandas as pd 
import numpy as np 
import math as mt
class Myknn :
    def __init__(self,dataframe_path:str):
        self.dataframepath = dataframe_path
        self.k_similar = 5
        self.training = 75
        self.test =15
        self.mode ='classification'

    def __setup(self):
        self.raw_dataframe = pd.read_csv(self.dataframepath)
        
 
    def __preprocess(self):
        self.preprocessed_dataframe = self.raw_dataframe.copy()

        binary_columns = ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']
        binary_mapping = {'yes': 1, 'no': 0}
        for column in binary_columns:
            self.preprocessed_dataframe[column] = self.preprocessed_dataframe[column].map(binary_mapping)

        gender_mapping = {"Male": 0, "Female": 1}
        self.preprocessed_dataframe['Gender'] = self.preprocessed_dataframe['Gender'].map(gender_mapping)

        non_binary_categorical_variables_columns = ['CAEC', 'MTRANS', 'CALC']
        for column in non_binary_categorical_variables_columns:
            dummies = pd.get_dummies(self.preprocessed_dataframe[column], prefix=column)
            self.preprocessed_dataframe.drop(column, axis=1, inplace=True)
            self.preprocessed_dataframe = pd.concat([self.preprocessed_dataframe, dummies], axis=1)

        numerical_columns_to_be_normalized = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        for column in numerical_columns_to_be_normalized:
            self.preprocessed_dataframe[column] = self.__normalize_column(self.preprocessed_dataframe[column])

        self.__split_data()

    def __normalize_column(self, col):
        return (col - col.min()) / (col.max() - col.min())
    
    def __split_data(self):
        totalrows = len(self.preprocessed_dataframe)
        trainingsize = int(totalrows/100)*self.training
        testsize =  int(totalrows/100)*self.test

        self.trainingDataset = self.preprocessed_dataframe[0:trainingsize]
        self.testDataset = self.preprocessed_dataframe[trainingsize:trainingsize + testsize]
    
    def __euclidiandistance(self,data,query):
        return mt.sqrt(((data - query)**2).sum())

    def __findneighbours(self,query_point,k=5):
        distances = []
        for index , row in self.preprocessed_dataframe.iterrows():
            distance=self.__euclidiandistance(row[:-1],query_point)
            distances.append((distance,index))
        distances.sort(key=lambda x : x[0])
        neightboors = distances[:k]
        return neightboors
    
    def predict(self, query_point, k=5):
        if self.mode == 'classification':
            neighbors = self.__findneighbours(query_point, k)
            labels = [self.preprocessed_dataframe.iloc[i[1]]['NObeyesdad'] for i in neighbors]  # Replace 'target_column_name' with your actual target column name
            prediction = max(set(labels), key=labels.count)
        return prediction
    
    def train(self):
        self.__setup()
        self.__preprocess()
        self.predict()
        return self.preprocessed_dataframe


if __name__ == "__main__":

    model = Myknn('ObesityDataSet.csv')
    
    print(model.train())
