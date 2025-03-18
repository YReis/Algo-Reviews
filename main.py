from modelreviews.data import dataprovider
from modelreviews.finalmodels import Myknn, MlLinearRegression, NeuralNet

datapath = 'datasets/RealState.csv'
dataprovider_instance = dataprovider(datapath, 70, 30)
traindata, testdata = dataprovider_instance.providedata('split')


ml_model = MlLinearRegression(traindata, testdata)
ml_model.trainanduse()

neural_net_model = NeuralNet(traindata, testdata, [20, 10])
neural_net_model.trainanduse()