from reviews.data import dataprovider
from reviews.finalmodels import Myknn

datapath = 'ObesityDataSet.csv'



dataprovider = dataprovider(datapath,70,30)
traindata,testdata=dataprovider.providedata()
model = Myknn(traindata,testdata)

print(model.trainadnuse())