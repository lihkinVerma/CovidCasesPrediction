
import random
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from Constants import CovidResourcesAndHyperparameters
from DeepLearningModel.ModelTraining import TrainNeuralNetwork
from DeepLearningModel.ModelTesting import TestNeuralNetwork

class DataProcessing:
    def __init__(self, dataToLearn = None, featuresToConsider = None, featureToPredict = None, nTimeSteps = 11, batchSize = 5, epochs=1000, lr = 1e-3):
        self.timeColumn = ['date']
        self.data = dataToLearn
        self.featuresToConsider = featuresToConsider if featuresToConsider is not None else CovidResourcesAndHyperparameters.featuresToConsider
        self.featureToPredict = featureToPredict if featureToPredict is not None else CovidResourcesAndHyperparameters.featureToPredict
        self.nTimeSteps = nTimeSteps  # nTimeStepsToConsiderInSequence
        self.batchSize = batchSize
        self.epochs = epochs
        self.lr = lr
        self.helper = Helper()

    def splitDataToSequences(self):
        dates = self.data[self.timeColumn]
        df = self.data[self.featuresToConsider]
        listOfDateInputOutput = [(dates.loc[i + self.nTimeSteps].tolist()[0],
                              df.loc[i : i + self.nTimeSteps - 1],
                              df[self.featureToPredict].loc[i + self.nTimeSteps]) if i + self.nTimeSteps < len(df) else None
                             for i in range(0, len(df))
                             ]
        return listOfDateInputOutput

    def splitSequencesForTrainingAndTesting(self):
        sequences = self.splitDataToSequences()
        train_size = int(0.8 * len(sequences))
        ll = list(range(len(sequences)))
        trainInd = random.sample(ll, train_size)
        testInd = list(set(ll) - set(trainInd))
        trainingSequences = list(map(sequences.__getitem__, trainInd))
        testingSequences = list(map(sequences.__getitem__, testInd))
        return (trainingSequences, testingSequences)

    def learnTrainingSequences(self, trainingSequences, modelNameToSave = None):
        modelNameToSave = modelNameToSave+'.pth' if modelNameToSave is not None else modelNameToSave
        trainNN = TrainNeuralNetwork.TrainNeuralNetwork(trainingSequences = trainingSequences,
                                                        featuresToConsider = self.featuresToConsider,
                                                        featureToPredict = self.featureToPredict,
                                                        nTimeSteps = self.nTimeSteps,
                                                        batchSize = self.batchSize,
                                                        epochs = self.epochs,
                                                        lr = self.lr)
        trainNN.trainModelForSequences()

        trainedModelName = trainNN.saveTorchModel(modelNameToSave)
        return trainedModelName

    def predictOnTestingSequences(self, testingSequences, trainedModelName):
        testNN = TestNeuralNetwork.TestNeuralNetwork(testingSequences)
        testNN.loadTorchModel(trainedModelName)
        testResults = testNN.testModelForSequences()
        return testResults

    def obtainOutputForAllSequences(self, maxPrediction, minPrediction, trainedModelName):
        sequences = self.splitDataToSequences()
        dff = self.predictOnTestingSequences(sequences, trainedModelName)
        dff.loc[:, ['actualCovidCases', 'predictedCovidCases']] = \
            dff.loc[:, ['actualCovidCases', 'predictedCovidCases']] * (maxPrediction - minPrediction) + minPrediction
        return dff

    def visualizeOutputForAllSequences(self, maxPrediction, minPrediction, trainedModelName):
        suffix = 'SeqLength-{} Epochs-{} Lr-{} Predicting-{}Cases'
        suffix = suffix.format(self.nTimeSteps, self.epochs, self.lr, self.featureToPredict)
        dff = self.obtainOutputForAllSequences(maxPrediction, minPrediction, trainedModelName)
        datesToPrintOnGraph = [i if i.endswith('-01') else None for i in dff.date.tolist()]
        self.helper.saveDataFrame(dff, 'OutputForWholeDataset'+suffix+'.csv')
        dff.loc[:, ['actualCovidCases', 'predictedCovidCases']].plot.line()
        plt.title('ModelPredictionForAllSequences\n' + suffix)
        plt.xticks(range(0, len(datesToPrintOnGraph)), datesToPrintOnGraph, rotation = 60)
        plt.xlabel('Timestamp')
        plt.ylabel('Number of Cases')
        plt.savefig(CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/ModelPredictionForAllSequences' + suffix + '.png',
                    dpi = 300, bbox_inches='tight')
        plt.show()
        return dff

    def splitSequencesForKfoldCrossValidation(self, kfold):
        sequences = self.splitDataToSequences()
        kf = KFold(n_splits=kfold, shuffle=True)
        splits = kf.split(sequences)
        kfoldSequencesBreakedup = []
        for trainInd, testInd in splits:
            try:
                trainInd, testInd = shuffle(trainInd), shuffle(testInd)
                trainingSequences = list(map(sequences.__getitem__, trainInd))
                testingSequences = list(map(sequences.__getitem__, testInd))
                kfoldSequencesBreakedup.append((trainingSequences, testingSequences))
            except Exception as e:
                pass
        return kfoldSequencesBreakedup

    def predictResultsForKfolds(self, kfold, maxPrediction, minPrediction):
        sequencesBreakedup = self.splitSequencesForKfoldCrossValidation(kfold)
        count = 1
        suffix = 'SeqLength-{} epochs-{} lr-{}'
        suffix = suffix.format(self.nTimeSteps, self.epochs, self.lr)
        listOfDataframes = []
        for trainingSequences, testingSequences in sequencesBreakedup:
            name = 'CovidMultivariateLSTMModel' + suffix + 'Kfold' + str(count)
            trainedModelName = self.learnTrainingSequences(trainingSequences, name)
            testResultsDataFrame = self.predictOnTestingSequences(testingSequences, trainedModelName)
            testResultsDataFrame.loc[:, ['actualCovidCases', 'predictedCovidCases']] = \
                testResultsDataFrame.loc[:, ['actualCovidCases', 'predictedCovidCases']] * (
                            maxPrediction - minPrediction) + minPrediction
            listOfDataframes.append(testResultsDataFrame)
            count = count + 1
        name = 'KfoldValidationPrediction ' + suffix
        self.helper.storeListOfDataframesOnSameExcel(listOfDataframes, name)
        return name + '.xlsx'

class Helper:
    def getDateInputOutputFromSequences(self, SequentialData):
        dates, input, output = [], [], []
        for inputDat in SequentialData:
            if inputDat is not None:
                d, i, o = inputDat
                dates.append(d)
                input.append(i.values)
                output.append(o)
        return (dates, input, output)

    def saveDataFrame(self, df, name):
        df.to_csv( CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/' +name, index = False)

    def storeListOfDataframesOnSameExcel(self, listOfDataframes, saveByName):
        writer = pd.ExcelWriter(CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/' +saveByName + '.xlsx', engine='xlsxwriter')
        count = 1
        for dataframe in listOfDataframes:
            dataframe.to_excel(writer, sheet_name='kfold-subset' + str(count), startrow=0, startcol=0)
            count = count + 1
        writer.save()

class PredictingFutureScenario:
    def __init__(self,
                 dataTilldate = None,
                 featuresToConsider=None,
                 modelNameForDeltaConfirmedPrediction = None,
                 modelNameForDeltaDeceasedPrediction = None,
                 modelNameForDeltaRecoveredPrediction = None,
                 modelNameForVaccinatedPrediction = None,
                 modelNameForVariantsPrediction = None):
        self.timeColumn = 'date'
        self.dataTilldate = dataTilldate
        self.featuresToConsider = featuresToConsider
        self.modelForDeltaConfirmedPrediction = TestNeuralNetwork.TestNeuralNetwork()
        self.modelForDeltaConfirmedPrediction.loadTorchModel(modelNameForDeltaConfirmedPrediction)
        self.modelForDeltaDeceasedPrediction = TestNeuralNetwork.TestNeuralNetwork()
        self.modelForDeltaDeceasedPrediction.loadTorchModel(modelNameForDeltaDeceasedPrediction)
        self.modelForDeltaRecoveredPrediction = TestNeuralNetwork.TestNeuralNetwork()
        self.modelForDeltaRecoveredPrediction.loadTorchModel(modelNameForDeltaRecoveredPrediction)
        self.modelForVaccinatedPrediction = TestNeuralNetwork.TestNeuralNetwork()
        self.modelForVaccinatedPrediction.loadTorchModel(modelNameForVaccinatedPrediction)
        self.modelForVariantsPrediction = TestNeuralNetwork.TestNeuralNetwork()
        self.modelForVariantsPrediction.loadTorchModel(modelNameForVariantsPrediction)

    def predictConfirmedCases(self, testSequence):
        self.modelForDeltaConfirmedPrediction.testData = testSequence
        preditedCases = self.modelForDeltaConfirmedPrediction.testModelForSequences()['predictedCovidCases'].tolist()[0]
        return preditedCases

    def predictDeceasedCases(self, testSequence):
        self.modelForDeltaDeceasedPrediction.testData = testSequence
        preditedCases = self.modelForDeltaDeceasedPrediction.testModelForSequences()['predictedCovidCases'].tolist()[0]
        return preditedCases

    def predictRecoveredCases(self, testSequence):
        self.modelForDeltaRecoveredPrediction.testData = testSequence
        preditedCases = self.modelForDeltaRecoveredPrediction.testModelForSequences()['predictedCovidCases'].tolist()[0]
        return preditedCases

    def predictVaccinatedCases(self, testSequence):
        self.modelForVaccinatedPrediction.testData = testSequence
        preditedCases = self.modelForVaccinatedPrediction.testModelForSequences()['predictedCovidCases'].tolist()[0]
        return preditedCases

    def predictVariantCases(self, testSequence):
        self.modelForVariantsPrediction.testData = testSequence
        preditedCases = self.modelForVariantsPrediction.testModelForSequences()['predictedCovidCases'].tolist()[0]
        return preditedCases

    def createLastDaysSequence(self, nTimeSteps):
        # dates = self.dataTilldate[self.timeColumn]
        df = self.dataTilldate[self.featuresToConsider]
        return (None, df.loc[-nTimeSteps:], None)

    def appendPredictionToOverallData(self):
        lastDate = self.dataTilldate['date'].tolist()[-1]
        lastDate = datetime.datetime.strptime(lastDate, '%Y-%m-%d')
        nextDate = lastDate + datetime.timedelta(days = 1)
        dayOfWeek = nextDate.weekday()
        nextDate = nextDate.strftime('%Y-%m-%d')
        newPredictedCases = {}
        lastDaysSequence = [self.createLastDaysSequence(11)]
        #print(lastDaysSequence)
        for feature in self.featuresToConsider:
            if feature=='delta7Confirmed':
                newPredictedCases.update({feature:self.predictConfirmedCases(lastDaysSequence)})
            if feature=='delta7Deceased':
                newPredictedCases.update({feature:self.predictDeceasedCases(lastDaysSequence)})
            if feature=='delta7Recovered':
                newPredictedCases.update({feature:self.predictRecoveredCases(lastDaysSequence)})
            if feature=='delta7Vaccinated':
                newPredictedCases.update({feature:self.predictVaccinatedCases(lastDaysSequence)})
            if feature=='variantsOfConcernReported':
                newPredictedCases.update({feature:self.predictVariantCases(lastDaysSequence)})
            if feature=='weekday':
                newPredictedCases.update({feature:dayOfWeek})
            print(newPredictedCases)
        newPredictedCases.update({self.timeColumn:nextDate})
        self.dataTilldate.append(newPredictedCases, ignore_index = True)
        print(self.dataTilldate.tail())

    def generateDataTillDate(self):
        pass