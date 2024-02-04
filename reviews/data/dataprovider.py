import pandas as pd 
import numpy as np

class dataprovider:
    def __init__(self,dataframepath,trainpercent,testpercent):
        self.dataframe = pd.read_csv(dataframepath)
        self.trainpercent = trainpercent
        self.testpercent = testpercent


    def __split_data(self):
        datasize = len(self.dataframe)
        trainingsize = int((datasize/100)*self.trainpercent)
        testsize =   int((datasize/100)*self.testpercent)
        print(trainingsize)
        print(testsize)
        trainingDataset = self.dataframe[0:trainingsize]
        testDataset = self.dataframe[trainingsize:trainingsize + testsize]
        return trainingDataset,testDataset
    def providedata(self,mode):
        trainingDataset,testDataset =self.__split_data()
        if(mode == 'split'):
            return trainingDataset,testDataset
        elif(mode == 'full'):
            return self.dataframe
        else:
            return trainingDataset,testDataset