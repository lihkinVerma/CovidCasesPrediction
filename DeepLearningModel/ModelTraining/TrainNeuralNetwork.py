
import torch
from DeepLearningModel import MultivariateLSTM, DataProcessing
from Constants import CovidResourcesAndHyperparameters

class TrainNeuralNetwork:
    def __init__(self, trainingSequences = None, featuresToConsider = None, featureToPredict = None, nTimeSteps = 11, batchSize = 5, epochs=1000, lr = 1e-3):
        self.trainingData = trainingSequences
        self.featuresToConsider = featuresToConsider if featuresToConsider is not None else CovidResourcesAndHyperparameters.featuresToConsider
        self.predicting = featureToPredict
        self.nFeatures = len(self.featuresToConsider)
        self.nTimeSteps = nTimeSteps  # nTimeStepsToConsiderInSequence
        self.batchSize = batchSize
        self.epochs = epochs
        self.lr = lr
        self.neuralNet = None
        self.criterion = None
        self.optimizer = None
        self.helper = DataProcessing.Helper()

    def instantiateMultiVariateLSTM(self):
        self.neuralNet = self.criterion = self.optimizer = None
        self.neuralNet = MultivariateLSTM.MV_LSTM(self.nFeatures, self.nTimeSteps)
        self.criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value
        self.optimizer = torch.optim.Adam(self.neuralNet.parameters(), lr = self.lr)

    def saveTorchModel(self, name=None):
        if name is None:
            suffix = 'SeqLength-{} epochs-{} lr-{} predicting={}'
            suffix = suffix.format(self.nTimeSteps, self.epochs, self.lr, self.predicting)
            name = 'CovidMultivariateLSTMModel'+ suffix + '.pth'
        torch.save(self.neuralNet, CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/' +name)
        return name

    def trainModelForSequences(self):
        self.instantiateMultiVariateLSTM()
        self.neuralNet.train()
        loss = None
        for t in range(self.epochs):
            for b in range(0, len(self.trainingData), self.batchSize):
                self.optimizer.zero_grad()
                inputData = self.trainingData[b: b + self.batchSize]
                dates, input, output = self.helper.getDateInputOutputFromSequences(inputData)
                x_batch = torch.tensor(input, dtype=torch.float32)
                y_batch = torch.tensor(output, dtype=torch.float32)
                self.neuralNet.init_hidden(x_batch.size(0))
                prediction = self.neuralNet(x_batch)
                loss = self.criterion(prediction.view(-1), y_batch)
                loss.backward()
                self.optimizer.step()
            if t % 500 == 0:
                try:
                    print('step : ', t, 'loss : ', loss.item())
                except Exception as e:
                    pass
        return