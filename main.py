from reviews.data import dataprovider
from reviews.finalmodels import Myknn,MlLinearRegression

datapath = 'RealState.csv'



dataprovider = dataprovider(datapath,70,30)
traindata,testdata=dataprovider.providedata()
model = MlLinearRegression(traindata,testdata)

model.trainanduse()