from reviews.data import dataprovider
from reviews.finalmodels import Myknn,MlLinearRegressor

datapath = 'RealState.csv'



dataprovider = dataprovider(datapath,70,30)
traindata,testdata=dataprovider.providedata()
model = MlLinearRegressor(traindata,testdata)

print(model.trainadnuse())