from ETL.DataLoader import DataLoader

import pandas as pd

loader = DataLoader()

returned = loader.TitanicLoader()

print(returned.dtypes)
print(returned.describe(), "\n")
