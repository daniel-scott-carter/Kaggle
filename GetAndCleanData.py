import datetime
import dateutil.parser as dateparser
import dateutil.relativedelta as rd
from bson import ObjectId
from pymongo import MongoClient
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re


def _format_rows(row):
    first_login = row["firstLogin"]

    if not isinstance(row["firstLogin"], datetime.datetime):
        first_login = datetime.datetime(row["firstLogin"])

    # datetime_object = datetime(first_login)
    plus_three = (first_login + rd.relativedelta(days=3))
    row["firstLoginPlusThree"] = plus_three

    return row


def _transform_rows(row):
    name = row["username"]
    if re.match("Guest[0-9]{7,}", name):
        row["playerIsGuest"] = 1
    else:
        row["playerIsGuest"] = 0

    email = row["email"]
    if email is not "" and email is not None:
        row["registeredEmail"] = 1
    else:
        row["registeredEmail"] = 0

    battlesLostPvE = row["battlesLostPvE"]
    if not isinstance(battlesLostPvE, int):
        print(battlesLostPvE)

    alliance = row["alliance"]
    if alliance is not "" and alliance is not None and alliance is not 0:
        row["playerInAlliance"] = 1

    FTUECompletes = row["ftueTypesCompleted"]
    if 313456751 in FTUECompletes:
        row["FTUEIntroCompleted"] = 1
    if 1427921910 in FTUECompletes:
        row["FTUEOnRailsCompleted"] = 1
    if 1484251237 in FTUECompletes:
        row["FTUEDayOneReturnIntroCompleted"] = 1
    if 673431887 in FTUECompletes:
        row["FTUEDayOneReturnCompleted"] = 1

    player_first_log_datetime = dateparser.parse(str(row["firstLogin_x"]))
    player_last_contact_datetime = dateparser.parse(str(row["lastContact"]))

    if (player_first_log_datetime + rd.relativedelta(days=3)) > player_last_contact_datetime:
        row["churnedWithin3Days"] = 1

    return row


def recode_empty_cells(dataframe, list_of_columns):
    for column in list_of_columns:
        dataframe[column] = dataframe[column].replace(r'\s+', np.nan, regex=True)
        dataframe[column] = dataframe[column].fillna(0)

    return dataframe


def _open_database_get_dataframe():
    # Specify the IP and port location of the MongoDB instance
    A_client = MongoClient("mongodb://35.163.233.188:27017/")
    SL_client = MongoClient("mongodb://52.38.222.249:27017/")

    # Which db in the instance are we interested in?
    A_db = A_client["Analytics-0000000008-KC3"]
    SL_db = SL_client["Analytics-0000000008-KC3"]
    print(A_db.name)

    # Which collection in the db are we interested in?
    A_player_database = A_db.lt_sl_playerDatabase
    A_sl_account_snapshots = A_db.lt_sl_account_snapshot
    A_sl_player_snapshots = A_db.lt_sl_player_snapshot
    SL_event_database = SL_db.serverEvents
    SL_sessions_collection = SL_db.lt_sessions

    print(A_player_database.name)

    # Query the whole playerdatabase collection
    players = A_player_database.find()
    print("Players: ", players.count())

    currentDate = datetime.datetime.now()
    one_months_ago = currentDate - rd.relativedelta(months=3)
    two_weeks_ago = currentDate - rd.relativedelta(weeks=5)

    acc_snapshot_agg_pipeline = [
        {
            "$match": {
                "$and": [{"date": {"$gt": one_months_ago}}, {"date": {"$lt": two_weeks_ago}}]
            }
        },
    ]

    player_snapshot_agg_pipeline = [
        {
            "$match": {
                "$and": [{"date": {"$gt": one_months_ago}}, {"date": {"$lt": two_weeks_ago}}]
            }
        },
    ]

    player_agg_pipeline = [
        {
            "$project": {
                "accountId": "$07accountId",
                "firstLogin": "$35firstLoginTime",
                "D0": "$39D0",
                "D1": "$40D1",
                "D2": "$41D2",
                "lastContact": "$36lastSessionEnd"
            }
        },
    ]

    # disconnect_agg_pipeline = [
    #     {
    #         "$match": {
    #             "$and": [{"event_name": "disconnected"}, {"evtm": {"$gt": one_months_ago}}]
    #         }
    #     },
    #
    #     {
    #         "$group": {
    #             "_id": "$account_id",
    #             "lastDisconnect": {
    #                 "$max": "$evtm"
    #             }
    #         }
    #     },
    # ]

    session_metrics_pipeline = [
        {
            "$match": {
                "$and": [{"session_start_time": {"$gt": one_months_ago}}, {"session_end_time": {"$lt": two_weeks_ago}}]
            }
        }
    ]

    banned_accounts_aggregation = [
        {
            "$project": {
                "length": {"$sum": [{"$size": "$devices"}, {"$size": "$accounts"}]},
                "accounts": 1
            }
        },
        {
            "$match": {
                "length": {"$gte": 8}
            }
        },
        {
            "$group": {
                "_id": 0,
                "accounts": {"$push": "$accounts"}
            }
        },
    ]

    # We must aggregate to get the cursor immediately before operating. This is because the cursor times out
    # after 10 minutes.
    print("")
    print("STEP 1 ================================================================")
    print("1.1 Get account snapshot aggregation")
    snapshots = A_sl_account_snapshots.aggregate(acc_snapshot_agg_pipeline)
    print("1.2 Generate Acc Snaps DF")
    account_snapshot = pd.DataFrame(list(snapshots))
    print("1.3 Drop un-needed columns")
    account_snapshot = account_snapshot.drop([
        "isSega",
        "accountLanguage",
        "accountPlatform"], axis=1)
    print("Generating Acc Snaps DF DONE!")

    print("")
    print("Step 1 Info: ")
    print("Account Snapshot shape: ", account_snapshot.shape)
    print("Unique account Ids in account snapshot: ", account_snapshot["accountId"].nunique())
    print("STEP 1 ================================================================")
    print("")

    print("STEP 2 ================================================================")
    print("2.1 Get player Snapshot aggregation")
    player_snaps = A_sl_player_snapshots.aggregate(player_snapshot_agg_pipeline)
    print("2.2 Generate Play Snaps DF")
    player_snapshot = pd.DataFrame(list(player_snaps))
    print("2.3 Drop un-needed columns")
    player_snapshot = player_snapshot.drop([
        "ftuePhaseType",
        "playerDescription"], axis=1)
    player_snapshot = player_snapshot.drop(player_snapshot.columns[40], axis=1)
    print("2.4 Recode Empty Cells for: pointsSupport, purchasedGoldCurrent, alliance")
    player_snapshot = recode_empty_cells(player_snapshot, ["pointsSupport", "purchasedGoldCurrent", "alliance"])
    print("Generating Play Snaps DF DONE!")

    print("")
    print("Step 2 Info: ")
    print("player Snapshot shape: ", player_snapshot.shape)
    print("Unique account Ids in player Snapshot", player_snapshot["accountId"].nunique())
    print("STEP 2 ================================================================")
    print("")

    print("STEP 3 ================================================================")
    print("3.1 Get player database aggregation")
    first_logins = A_player_database.aggregate(player_agg_pipeline)
    print("3.2 Generate player database DF")
    logs = pd.DataFrame(list(first_logins))
    print("3.3 Drop players with no lastContact")
    print("Unique accountIds in playerDatabase DF info before drop no lastContact: ", logs["accountId"].nunique())
    logs = logs.loc[logs["lastContact"] != ""]
    print("Unique accountIds in playerDatabase DF after drop no lastContact: ", logs["accountId"].nunique())
    print("--------- Generating Login DF DONE!")

    print("STEP 3 ================================================================")
    print("")

    print("STEP 4 ================================================================")
    print("4.1 Merging Acc_snap and playerDatabase")
    print("Uniques acc snap before merge: ", account_snapshot["accountId"].nunique())
    account_snapshot = pd.merge(account_snapshot, logs, left_on="accountId", right_on="accountId", how='inner')
    print("Uniques acc snap after merge: ", account_snapshot["accountId"].nunique())
    print("4.2 Drop account snapshots with no first login")
    account_snapshot = account_snapshot.loc[account_snapshot["firstLogin"] != ""]
    print("Uniques acc snap after drop no first login: ", account_snapshot["accountId"].nunique())
    print("4.3 Drop additional Id column")
    account_snapshot = account_snapshot.drop(["_id_y"], axis=1)
    print("---------Done Merging")

    print("")
    print("Step 4 Info: ")
    print("New Acc Snapshot Shape", account_snapshot.shape)
    print("STEP 4 ================================================================")
    print("")

    print("STEP 5 ================================================================")
    print("5.1 Calculate firstLoginPlusThree for each player/account")
    account_snapshot["firstLoginPlusThree"] = 0
    account_snapshot = account_snapshot.apply(_format_rows, broadcast=True, reduce=False, axis=1)
    print("5.2 Rename Date column for the snapshot")
    account_snapshot = account_snapshot.rename(index=str, columns={"date": "accountSnapDate"})

    print("5.3 Drop all snapshots which are NOT on the 3rd Day after player first logged in")
    print("Uniques acc snap before drop non-3Days: ", account_snapshot["accountId"].nunique())
    account_snapshot = account_snapshot.loc[account_snapshot["accountSnapDate"].dt.date ==
                                            account_snapshot["firstLoginPlusThree"].dt.date]
    print("Uniques acc snap after drop non-3Days: ", account_snapshot["accountId"].nunique())
    print("5.4 Calculate time difference between last snapshot and actual 3 Day end time")
    account_snapshot["accountSnapshotAndLoginPlus3TimeDifference"] = 0
    account_snapshot["accountSnapshotAndLoginPlus3TimeDifference"] = account_snapshot["firstLoginPlusThree"] - \
                                                                     account_snapshot["accountSnapDate"]
    print("5.5 Drop all duplicate snapshots by accountId. Keep only the last one.")
    account_snapshot = account_snapshot.drop_duplicates('accountId', keep="last")

    print("")
    print("Step 5 Info:")
    print("New Acc Snapshot Shape", account_snapshot.shape)
    print("Uniques acc snap after drop duplicates: ", account_snapshot["accountId"].nunique())
    print("STEP 5 ================================================================")
    print("")

    print("STEP 6 ================================================================")
    print("6.1 Merge player_snapshot and playerDatabase")
    player_snapshot = pd.merge(player_snapshot, logs, left_on="accountId", right_on="accountId", how='inner')
    print("Uniques acc snap after merge: ", player_snapshot["accountId"].nunique())
    print("6.2 Drop all player_snapshots with no firstLogin")
    player_snapshot = player_snapshot.loc[player_snapshot["firstLogin"] != ""]
    print("Uniques acc snap after drop: ", player_snapshot["accountId"].nunique())
    print("6.3 Initialise firstLoginPlusThree for each player")
    player_snapshot["firstLoginPlusThree"] = 0

    print("6.4 Calculate firstLoginPlusThree for each player")
    player_snapshot = player_snapshot.apply(_format_rows, broadcast=True, reduce=False, axis=1)

    print("6.5 Rename Date Column")
    player_snapshot = player_snapshot.rename(index=str, columns={"date": "playerSnapDate"})

    print("6.6 Drop all snapshots which are NOT on the 3rd day after player firstLogin")
    player_snapshot = player_snapshot.loc[player_snapshot["playerSnapDate"].dt.date ==
                                          player_snapshot["firstLoginPlusThree"].dt.date]
    print("Uniques acc snap after 3day drop: ", player_snapshot["accountId"].nunique())

    print("6.6 Calculate time difference between last snapshot and actual 3Day after firstLogin")
    player_snapshot["playerSnapshotAndLoginPlus3TimeDifference"] = 0
    player_snapshot["playerSnapshotAndLoginPlus3TimeDifference"] = player_snapshot["firstLoginPlusThree"] - \
                                                                   player_snapshot["playerSnapDate"]

    print("6.7 Drop duplicate snapshots by accountId, keep only last snapshot")
    player_snapshot = player_snapshot.drop_duplicates('accountId', keep="last")

    print("")
    print("Step 6 Info:")
    print("New Player Snapshot Shape", player_snapshot.shape)
    print("Uniques acc snap after duplicate drop: ", player_snapshot["accountId"].nunique())
    print("STEP 6 ================================================================")
    print("")

    print("STEP 7 ================================================================")
    print("7.1 Merge acc_snap and play_snap")
    df = pd.merge(account_snapshot, player_snapshot, left_on="accountId", right_on="accountId", how="inner")
    print("Uniques after merge: ", player_snapshot["accountId"].nunique())
    print("7.2 Rename lastContact column")
    df = df.rename(index=str, columns={"lastContact_x": "lastContact"})
    print("7.3 Drop duplicate columns")
    df = df.drop([
        "_id_x_x",
        "_id_y",
        "D0_y",
        "D1_y",
        "D2_y",
        "firstLogin_y",
        "firstLoginPlusThree_y",
        "_id_x_y",
        "timezoneOffset",
        "timezoneOffsetMinutes",
        "lastStateChange",
        "accountRegion",
        "playerName",
        "heroState",
        "heroAvatar",
        "deviceId",
        "lastContact_y"], axis=1)

    df.D0_x = df.D0_x.astype(int)
    df.D1_x = df.D1_x.astype(int)
    df.D2_x = df.D2_x.astype(int)
    df.isVip = df.isVip.astype(int)

    print("")
    print("Step 7 Info: ")
    print("DF Shape: ", df.shape)
    print("STEP 7 ================================================================")
    print("")

    # last_disconnects = SL_event_database.aggregate(disconnect_agg_pipeline)
    # last_disconnects = pd.DataFrame(list(last_disconnects))
    #
    # last_disconnects = last_disconnects.loc[last_disconnects["_id"] != ""]
    #
    # df = pd.merge(df, last_disconnects, left_on="accountId", right_on="_id", how="inner")
    #
    # df = df.drop(["_id"], axis=1)

    # merge session stuff here, then drop unnecessary columns

    print("STEP 8 ================================================================")
    print("8.1 Get Session Database aggregation")
    session_data = SL_sessions_collection.aggregate(session_metrics_pipeline)
    print("8.2 Create session dataframe")
    sessions = pd.DataFrame(list(session_data))
    print("8.3 Drop un-needed columns")
    sessions = sessions.drop(["_id",
                              "castle_level",
                              "session_dow",
                              "session_hod",
                              "session_id",
                              "week"], axis=1)
    print("8.4 Merge the session data into Df using playerId")
    print("Uniques before merge: ", df["accountId"].nunique())
    df = pd.merge(df, sessions, left_on="playerId", right_on="player_id")
    print("Uniques after merge: ", df["accountId"].nunique())

    print("8.5 Get only sessions in the 3 day range for each player")
    df = df.loc[df["session_start_time"].dt.date >= df["firstLogin_x"].dt.date]
    df = df.loc[df["session_end_time"].dt.date <= df["firstLoginPlusThree_x"].dt.date]

    print("8.6 Calculate totals and averages from Session Data")
    df = df.join(df.groupby(df["player_id"])["session_events"].sum(), on="player_id", rsuffix="_total")
    df = df.join(df.groupby(df["player_id"])["session_events"].mean(), on="player_id", rsuffix="_avg")
    df = df.join(df.groupby(df["player_id"])["session_length_mins"].sum(), on="player_id", rsuffix="_total")
    df = df.join(df.groupby(df["player_id"])["session_length_mins"].mean(), on="player_id", rsuffix="_avg")
    df = df.join(df.groupby(df["player_id"])["avg_event_interval"].mean(), on="player_id", rsuffix="_avg")
    print("8.7 Drop un-needed columns")
    df = df.drop(["session_end_time",
                  "session_start_time",
                  "player_id",
                  "session_events",
                  "session_length_mins",
                  "avg_event_interval"], axis=1)
    print("8.8 Drop duplicate rows on accountId")
    print("Uniques before drop: ", df["accountId"].nunique())
    df = df.drop_duplicates(subset='accountId', keep="last")
    print("Uniques after drop: ", df["accountId"].nunique())

    print("STEP 8 ================================================================")
    print("")

    print("STEP 9 ================================================================")
    print("9.1 Initialise Engineered Features")

    df["playerIsGuest"] = 0
    df["registeredEmail"] = 0
    df["playerInAlliance"] = 0
    df["battlesWon"] = df["battlesWonPvE"] + df["battlesWonPvP"]
    df["battlesLost"] = df["battlesLostPvE"] + df["battlesLostPvP"]
    df["totalBattles"] = df["battlesWonPvE"] + df["battlesWonPvP"] + df["battlesLostPvE"] + df["battlesLostPvP"]
    df["FTUEIntroCompleted"] = 0
    df["FTUEOnRailsCompleted"] = 0
    df["FTUEDayOneReturnIntroCompleted"] = 0
    df["FTUEDayOneReturnCompleted"] = 0
    df["churnedWithin3Days"] = 0

    print("9.2 Apply Row Transformations/Cleaning/Feature Engineering")
    df = df.apply(_transform_rows, broadcast=True, reduce=False, axis=1)

    print("9.3 Save Raw Data to CSV")
    df.to_csv("RawData.csv", index=False, encoding="utf-8")

    print("9.4 Drop qualitative or useless columns for ML")
    df = df.drop([
        "username",
        "accountId",
        "accountSnapDate",
        "email",
        "D0_x",
        "D1_x",
        "D2_x",
        "firstLoginPlusThree_x",
        "firstLogin_x",
        "accountSnapshotAndLoginPlus3TimeDifference",
        "alliance",
        "playerSnapDate",
        "deactivatedAtTime",
        "ftueTypesCompleted",
        "playerId",
        "playerSnapshotAndLoginPlus3TimeDifference",
        "lastContact",
        "purchasedGoldCurrent"], axis=1)

    # Now we have usable data!

    print("9.5 Save usable data to CSV")
    df.to_csv("Data.csv", index=False, encoding="utf-8")

    print(df)

    return df


dframe = _open_database_get_dataframe()

print("We are done.")
