import pandas as pd
import re

class DataLoader:

    variable = "I'm a variable"
    excluded_titles = None

    def exploreDataframe(self, dataframe):
        print("Info: ")
        print(dataframe.info())

        print("Column Data Types: ")
        print(dataframe.dtypes)

        print("Sample: ")
        print(dataframe.sample(20))

        print("How many nulls? ")
        print(dataframe.isnull().sum())

        print("Describe the data: ")
        print(dataframe.describe(include = "all"))

    def _correctColumnTypes(self, dataframe):

        dataframe["Cabin"] = dataframe["Cabin"].astype(str)

        return dataframe

    def _fillNACols(self, df):
        df['Age'].fillna(df['Age'].median(), inplace=True)

        # complete embarked with mode
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        # complete missing fare with median
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        return df

    def _initialiseRows(self, dataframe):


        # Calculate each persons entourage/family size (+1 to include the person themself)
        if not "EntourageSize" in dataframe.columns:
            dataframe["EntourageSize"] = dataframe["SibSp"] + dataframe["Parch"] + 1

        if not "IsALone" in dataframe.columns:
            dataframe["IsAlone"] = 1
            dataframe["IsAlone"].loc[dataframe["EntourageSize"] > 1] = 0

        if not "Title" in dataframe.columns:
            dataframe["Title"] = dataframe["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
            dataframe["Title"] = dataframe["Title"].replace("Mlle", "Miss")
            dataframe["Title"] = dataframe["Title"].replace("Ms", "Miss")
            dataframe["Title"] = dataframe["Title"].replace("Mme", "Mrs")

        if not "RoomSide" in dataframe.columns:
            dataframe["RoomSide"] = "Unknown"

        if not "Deck" in dataframe.columns:
            dataframe["Deck"] = "Unknown"

        return dataframe


    def _formatRows(self, row):

        if row["Sex"] is "male":
            row["Sex"] = 1
        else:
            row["Sex"] = 0

        # Drop any rows where embarked is 0 later
        if row["Embarked"] is not "" and row["Embarked"] is not None and row["Embarked"]:
            row["Embarked"] = 1
        else:
            row["Embarked"] = 0

        # Get only the cabin level (A, B, C etc) where n = null (filter later)
        if row["Cabin"] is not "" and row["Cabin"] is not None:
            cabinString = row["Cabin"]

            if cabinString[:1].isalpha():
                row["Deck"] = cabinString[:1]

            number = re.findall(r'^\D*(\d+)', cabinString)


            if not self._checkEmptyList(number):
                number = int(number[0])

                if number % 2 == 0:
                    row["RoomSide"] = "Port"
                else:
                    row["RoomSide"] = "Starboard"

        # use excludedTitles to replace any Titles with a freq < 5 with 'Other'
        if self.excluded_titles.loc[row["Title"]] == True:
            row["Title"] = "Other"


        return row

    def _checkEmptyList(self, theList):
        if not theList:
            return True
        else:
            return False


    def TitanicLoader(self, inputPath):
        df = pd.read_csv(inputPath, infer_datetime_format=True, encoding="utf-8")

        print("---- Initial exploration: ----")
        self.exploreDataframe(df)

        df = self._fillNACols(df)
        df = self._correctColumnTypes(df)
        df = self._initialiseRows(df)

        self.excluded_titles = (df["Title"].value_counts() < 5)
        df = df.apply(self._formatRows, broadcast=True, reduce=False, axis=1)

        print("---- Post Feature Engineering exploration: ----")
        print(df['Title'].value_counts())
        self.exploreDataframe(df)

        return df

