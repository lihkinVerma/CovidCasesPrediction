
import requests
import pandas as pd
from datetime import datetime
from Constants import CovidResourcesAndHyperparameters

class CollectCovidData:
    def __init__(self):
        self.cases = ['delta', 'delta7', 'total']
        self.stages = ['confirmed', 'deceased', 'recovered', 'vaccinated']
        self.covidCasesUrl = CovidResourcesAndHyperparameters.covidCasesUrl
        self.variantsOfConcernDataUrl = CovidResourcesAndHyperparameters.variantsOfConcernDataUrl
        self.covidData = None
        self.variantsOfConcernData = None

    # Data of Covid Cases changing Regularly
    def getCovidData(self):
        response = requests.get(self.covidCasesUrl)
        self.covidData = response.json()

    def getVariantsOfConcernData(self):
        response = requests.get(self.variantsOfConcernDataUrl)
        self.variantsOfConcernData = response.json()

    def getStagewiseData(self, dataCaseWise, stage):
        if dataCaseWise is None:
            return None
        if stage == 'confirmed':
            return dataCaseWise.get('confirmed')
        if stage == 'deceased':
            return dataCaseWise.get('deceased')
        if stage == 'recovered':
            return dataCaseWise.get('recovered')
        if stage == 'tested':
            return dataCaseWise.get('tested')
        if stage == 'vaccinated':
            return dataCaseWise.get('vaccinated')
        return None

    def getCasewiseData(self, overalldata, param):
        if param == 'delta':
            return overalldata.get('delta')
        if param == 'delta7':
            return overalldata.get('delta7')
        if param == 'total':
            return overalldata.get('total')
        return None

    # Data regarding Variants of Concern
    def getVariantCasesReportedFromIndia(self):
        allIndianCases = []
        for variantName, variantData in self.variantsOfConcernData.items():
            countryCounts = variantData.get('Country counts')
            for countryCases in countryCounts:
                if countryCases.get('country') == 'India':
                    allIndianCases = allIndianCases + countryCases.get('counts')
        return allIndianCases

    def convertAllIndiaCasesToDataframe(self, allIndianCases):
        df = pd.DataFrame(allIndianCases)
        aggregation_functions = {'count': 'sum'}
        df = df.groupby(df['date']).aggregate(aggregation_functions)
        df = df.rename(columns={'count': 'variantsOfConcernReported'})
        df['date'] = df.index
        df.index = range(len(df))
        return df

    def convertStatewiseDataToDataframe(self, statewisedata):
        df = pd.DataFrame(statewisedata)
        df.columns = ['overalldata']
        for case in self.cases:
            df[case] = df.overalldata.apply(self.getCasewiseData, args=[case])
            for stage in self.stages:
                df[case + stage.title()] = df[case].apply(self.getStagewiseData, args=[stage])
        df['date'] = df.index
        df.index = range(len(df))
        return df

    # Some Helper Function
    def rearrangeColumns(self, df):
        columnsInOrder = ['date']
        for case in self.cases:
            for stage in self.stages:
                columnsInOrder.append(case + stage.title())
        return df[columnsInOrder]

    def saveDataInCSV(self, nameOfCSV, df):
        df = df.fillna(0)
        df.to_csv(nameOfCSV + '.csv', index=False)

    def appendDayOfWeek(self, df):
        dates = df.date
        weekdays = [datetime.strptime(date, '%Y-%m-%d').weekday() for date in dates]
        df['weekday'] = weekdays
        return df

    # Obtain Data Region Wise
    def getDelhidata(self, save = False):
        self.getCovidData()
        df = self.convertStatewiseDataToDataframe(self.covidData.get('DL'))
        df = self.rearrangeColumns(df)
        df = self.appendDayOfWeek(df)
        self.getVariantsOfConcernData()
        indianCases = self.getVariantCasesReportedFromIndia()
        df_variants = self.convertAllIndiaCasesToDataframe(indianCases)
        df = pd.merge(df, df_variants, how='left', on='date')
        df = df.fillna(0)
        if save:
            self.saveDataInCSV(CovidResourcesAndHyperparameters.pathToSavePredictionRelatedData + "/DelhiCovidData", df)
        return df
