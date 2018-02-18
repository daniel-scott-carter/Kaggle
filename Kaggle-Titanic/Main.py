from ETL.DataLoader import DataLoader

import pandas as pd

loader = DataLoader()

returned = loader.TitanicLoader('CSVData\\train.csv')


print(returned.dtypes)
print(returned.describe(), "\n")
