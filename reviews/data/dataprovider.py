import pandas as pd 
import numpy as np

class dataprovider:
    def __init__(self,dataframepath,trainpercent,testpercent):
        self.dataframepath = dataframepath
        self.dataframe = pd.read_csv(dataframepath)
        self.trainpercent = trainpercent
        self.testpercent = testpercent


    def __split_data(self):
        totalrows = len(self.dataframe)
        trainingsize = int(totalrows/100)*self.trainpercent
        testsize =  int(totalrows/100)*self.testpercent

        self.trainingDataset = self.preprocessed_dataframe[0:trainingsize]
        self.testDataset = self.preprocessed_dataframe[trainingsize:trainingsize + testsize]