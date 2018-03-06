from ETL.DataLoader import DataLoader
from Ensemble.ClassifierEnsemble import ClassifierEnsemble

import pandas as pd
import numpy as np

loader = DataLoader()
ensemble = ClassifierEnsemble()

dataframe = loader.TitanicLoader('CSVData\\train.csv')

msk = np.random.rand(len(dataframe)) < 0.85

training_set = dataframe[msk]
test_set = dataframe[~msk]

# print(dataframe.dtypes)
# print(dataframe.describe(), "\n")

Target = ["Survived"]

index = np.argwhere(dataframe.columns.values == "Survived")
Predictors = np.delete(dataframe.columns.values, index)

if __name__ == '__main__':
    ensemble.trainAllClassifiers(training_set, Predictors, Target)

    #ensemble.getAndScoreVotingEnsemble(dataframe, Predictors, Target, "hard")

    ensemble.getAllClassifierPredictions(test_set, Predictors, Target)








