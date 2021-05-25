#----------------------------------------------------
# Covid 19 cases prediction
# Learning Sequences
# Developer: Nikhil Verma
# Funding agency: IIT Delhi, Delhi, India
# Motif: Research
#----------------------------------------------------

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Constants import CovidResourcesAndHyperparameters
from DataCollection import CollectCovidData
from DeepLearningModel import DataProcessing

if __name__ == "__main__":
	dataCollection = CollectCovidData.CollectCovidData()
	df = dataCollection.getDelhidata(True)
	#df = pd.read_csv(CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + '/DelhiCovidData.csv')
	# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

	# Defining Learning Constants
	featuresToConsider = CovidResourcesAndHyperparameters.featuresToConsider
	featureToPredict =  CovidResourcesAndHyperparameters.featureToPredict
	nTimeSteps = CovidResourcesAndHyperparameters.nTimeStepsInSequence
	epochs = CovidResourcesAndHyperparameters.epochs
	kfold = CovidResourcesAndHyperparameters.kfold
	learningRate = CovidResourcesAndHyperparameters.learningRate
	batchSize = CovidResourcesAndHyperparameters.batchSize

	# Doing Normalisation
	statsOfAllColumns = df.describe()
	maxPrediction, minPrediction = df[featureToPredict].max(), df[featureToPredict].min()
	scaler = MinMaxScaler(feature_range=(0, 1))
	for feature in featuresToConsider:
		featureData = df[feature]
		normalizedData = scaler.fit_transform(featureData.values.reshape(-1, 1))
		df[feature] = normalizedData.flatten()

	predictFuture = DataProcessing.PredictingFutureScenario(df, featuresToConsider,
															CovidResourcesAndHyperparameters.a,
															CovidResourcesAndHyperparameters.b,
															CovidResourcesAndHyperparameters.c,
															CovidResourcesAndHyperparameters.d,
															CovidResourcesAndHyperparameters.e,
															)
	# df = predictFuture.appendPredictionToOverallData(maxPrediction, minPrediction)
	predictForFutureDelta = 7
	df = predictFuture.generateDataForNextDeltaDays(predictForFutureDelta)
	predictFuture.visualizeDataPredictedForFuture(statsOfAllColumns, predictForFutureDelta)
	'''
	# Processing Data
	dataProcessing = DataProcessing.DataProcessing(dataToLearn = df,
												   featuresToConsider = featuresToConsider,
												   featureToPredict = featureToPredict,
												   nTimeSteps = nTimeSteps,
												   batchSize = batchSize,
												   epochs = epochs,
												   lr = learningRate)
	trainingSequences, testingSequences = dataProcessing.splitSequencesForTrainingAndTesting()

	# Training and Testing Sequences
	trainedModelName = dataProcessing.learnTrainingSequences(trainingSequences)
	testResultsDataFrame = dataProcessing.predictOnTestingSequences(testingSequences, trainedModelName)
	print(testResultsDataFrame)
	dataProcessing.visualizeOutputForAllSequences(maxPrediction, minPrediction, trainedModelName)

	# Cross Validation of Model
	# dataProcessing.predictResultsForKfolds(kfold, maxPrediction, minPrediction)
	'''
