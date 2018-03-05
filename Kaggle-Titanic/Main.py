from ETL.DataLoader import DataLoader
from Ensemble.ClassifierEnsemble import ClassifierEnsemble

import pandas as pd
import numpy as np

loader = DataLoader()
ensemble = ClassifierEnsemble()

dataframe = loader.TitanicLoader('CSVData\\train.csv')

print(dataframe.dtypes)
print(dataframe.describe(), "\n")

Target = ["Survived"]

index = np.argwhere(dataframe.columns.values == "Survived")
Predictors = np.delete(dataframe.columns.values, index)

if __name__ == '__main__':
    ensemble.trainAllClassifiers(dataframe, Predictors, Target)

    ensemble.getAndScoreVotingEnsemble(dataframe, Predictors, Target, "hard")








