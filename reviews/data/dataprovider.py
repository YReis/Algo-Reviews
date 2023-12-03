import pandas as pd 
import numpy as np

class dataprovider:
    def __init__(self,dataframepath,trainpercent,testpercent):
        self.dataframepath = dataframepath
        self.dataframe = pd.read_csv(dataframepath)
        self.trainpercent = trainpercent
        self.testpercent = testpercent


    def __split_data(self):
        trainingsize = self.trainpercent
        testsize =   self.testpercent

        trainingDataset = self.dataframe[0:trainingsize]
        testDataset = self.dataframe[trainingsize:trainingsize + testsize]
        return trainingDataset,testDataset
    def providedata(self):
        trainingDataset,testDataset =self.__split_data()
        return trainingDataset,testDataset