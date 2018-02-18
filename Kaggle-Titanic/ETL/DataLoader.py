import pandas as pd

class DataLoader:

    variable = "I'm a variable"

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
            #title_names = (dataframe["Title"].value_counts() < 10)
            #dataframe = dataframe["Title"].apply(lambda x: "Other" if title_names.loc[x] == True else x)



        # Commented out because: should be one-hot encoded, no need to do by hand
        # if not "EmbarkedQ" in dataframe.columns:
        #     dataframe["EmbarkedQ"] = 0
        #
        # if not "EmbarkedC" in dataframe.columns:
        #     dataframe["EmbarkedC"] = 0
        #
        # if not "EmbarkedS" in dataframe.columns:
        #     dataframe["EmbarkedS"] = 0

        return dataframe



    def _formatRows(self, row):

        if row["Sex"] is "male":
            row["Sex"] = 1
        else:
            row["Sex"] = 0

        # # Assign correct values to embarked columns
        # if row["Embarked"] is "Q":
        #     row["EmbarkedQ"] = 1
        #
        # if row["Embarked"] is "C":
        #     row["EmbarkedC"] = 1
        #
        # if row["Embarked"] is "S":
        #     row["EmbarkedS"] = 1

        # Drop any rows where embarked is 0 later
        if row["Embarked"] is not "" and row["Embarked"] is not None:
            row["Embarked"] = 1
        else:
            row["Embarked"] = 0

        if row["Cabin"] is not "" and row["Cabin"] is not None:

            cabinString = row["Cabin"]
            cabinString = cabinString.str[:1]
            row["Cabin"] = cabinString


        return row


    def TitanicLoader(self, inputPath):
        df = pd.read_csv(inputPath, infer_datetime_format=True, encoding="utf-8")

        print("---- Initial exploration: ----")
        self.exploreDataframe(df)

        df = self._fillNACols(df)
        df = self._initialiseRows(df)
        df = df.apply(self._formatRows, broadcast=True, reduce=False, axis=1)

        print("---- Post Feature Engineering exploration: ----")
        self.exploreDataframe(df)

        return df






    def AnotherMethod(self):
        print("GoodBye")
