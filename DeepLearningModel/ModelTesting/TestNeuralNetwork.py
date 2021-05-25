
import torch
import pandas as pd
from DeepLearningModel import MultivariateLSTM, DataProcessing
from Constants import CovidResourcesAndHyperparameters

class TestNeuralNetwork:
    def __init__(self, testSequences = None):
        self.neuralNet = None
        self.testData = testSequences
        self.helper = DataProcessing.Helper()

    def loadTorchModel(self, name):
        self.neuralNet = torch.load(CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/' + name)

    def testModelForSequences(self):
        dates, input, output = self.helper.getDateInputOutputFromSequences(self.testData)
        x_batch = torch.tensor(input, dtype=torch.float32)
        prediction = None
        if self.neuralNet is not None:
            self.neuralNet.init_hidden(x_batch.size(0))
            self.neuralNet.eval()
            prediction = self.neuralNet(x_batch)
            prediction = prediction.view(-1)
        df = pd.DataFrame(columns=['date', 'actualCovidCases', 'predictedCovidCases'])
        df.date = dates
        df.actualCovidCases = output
        df.predictedCovidCases = prediction.tolist() if prediction is not None else None
        return df