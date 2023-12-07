from data.dataprovider import dataprovider
datapath = 'RealState.csv'
dataprovider = dataprovider(datapath, 70, 30)
traindata, testdata = dataprovider.providedata()
print(traindata)