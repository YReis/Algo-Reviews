import pandas as pd 
import numpy as np 

class Myknn :
    def __init__(self,dataframe_path:str):
        self.dataframepath = dataframe_path

    def __setup(self,dataframe_path:str):
        self.raw_dataframe = pd.read_csv(dataframe_path)
        
    def __preprocess(self):
        dataframe_columns=self.raw_dataframe
    

    
    def train(self):
        self.__setup(self.dataframepath)
        return self.raw_dataframe


if __name__ == "__main__":

    model = Myknn('ObesityDataSet.csv')
    
    print(model.train())
