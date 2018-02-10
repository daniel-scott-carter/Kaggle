import pandas as pd

class DataLoader:

    variable = "I'm a variable"

    def _format_rows(self, row):

        if row["Sex"] is "male":
            row["Sex"] = 1
        else:
            row["Sex"] = 0

        if row["Embarked"] is not "" and row["Embarked"] is not None:
            row["Embarked"] = 1
        else:
            row["Embarked"] = 0

        return row


    def TitanicLoader(self):
        df = pd.read_csv('CSVData\\train.csv', infer_datetime_format=True, encoding="utf-8")

        df = df.apply(self._format_rows, broadcast=True, reduce=False, axis=1)

        return df






    def AnotherMethod(self):
        print("GoodBye")
