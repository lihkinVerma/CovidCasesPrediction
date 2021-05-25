
#---------------------------------------------
# Covid Related Sites to collect Data
#---------------------------------------------
covidCasesUrl = 'https://api.covid19india.org/v4/min/timeseries.min.json'
variantsOfConcernDataUrl = 'https://raw.githubusercontent.com/cov-lineages/lineages-website/master/_data/lineage_data.json'
## Other meaningful sites
#  https://www.gisaid.org/hcov19-variants/
#  https://nextstrain.org/community/banijolly/Phylovis/COVID-India?d=tree,frequencies,entropy&p=full
#  https://cov-lineages.org/lineages/lineage_A.html

#---------------------------------------------
# HyperParameters For Deep Learning Model
#---------------------------------------------
nTimeStepsInSequence = 11
epochs = 1000
kfold = 5
learningRate = 1e-5
batchSize = 10
featureToPredict = None
featureToPredict = 'delta7Confirmed'
featuresToConsider = ['delta7Confirmed', 'delta7Deceased', 'delta7Recovered',
                      'delta7Vaccinated', 'weekday', 'variantsOfConcernReported']

#---------------------------------------------
# Path to save Data generated
#---------------------------------------------
pathToSavePredictionRelatedData = 'SaveModelAndPredictionResults'
a = "CovidMultivariateLSTMModelSeqLength-11 epochs-1000 lr-1e-05 predicting=delta7Confirmed.pth"
b = "CovidMultivariateLSTMModelSeqLength-11 epochs-1000 lr-1e-05 predicting=delta7Deceased.pth"
c = "CovidMultivariateLSTMModelSeqLength-11 epochs-1000 lr-1e-05 predicting=delta7Recovered.pth"
d = "CovidMultivariateLSTMModelSeqLength-11 epochs-1000 lr-1e-05 predicting=delta7Vaccinated.pth"
e = "CovidMultivariateLSTMModelSeqLength-11 epochs-1000 lr-1e-05 predicting=variantsOfConcernReported.pth"