from datetime import datetime
import dateutil.relativedelta as rd
import dateutil.parser as dateparser
from bson import ObjectId
from pymongo import MongoClient
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re

# ------------------------------------------------------------------------------------
# Import Classifier files
import Classifiers

# TODO wrap ALL of this file up into functions, with one 'main' function to control all the crazy shizz that's going on in 'ere

# ------------------------------------------------------------------------------------
# Open up the database and get the good stuff


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
    SL_event_database = SL_db.serverEvents
    print(A_player_database.name)

    # Query the whole playerdatabase collection
    players = A_player_database.find()
    print(players.count())

    currentDate = datetime.now()
    two_months_ago = currentDate - rd.relativedelta(months=2)

    # This is a mongoDB aggregation pipeline
    agg_pipeline = [
        {
            "$match": {
                "$and": [{"event_name": "disconnected"}, {"evtm": {"$gt": two_months_ago}}]
            }
        },

        {
            "$group": {
                "_id": "$player_id",
                "lastDisconnect": {
                    "$last": "$evtm"
                }
            }
        },
    ]

    print("Aggregating")
    last_disconnects = SL_event_database.aggregate(agg_pipeline)
    print("Aggregation Done!")

    print("Generating DFs")
    # Create the dataframes we'll be working with
    df = pd.DataFrame(list(players))
    disconnects = pd.DataFrame(list(last_disconnects))
    print("Generating DFs done")

    print("Merging dfs...")

    df = pd.merge(df, disconnects, left_on='02playerId', right_on='_id', how='inner')

    print("Merging Done!")

    # Open up the database and get the right stuff
    # ------------------------------------------------------------------------------------
    # Delete useless columns, create new empty columns
    print("Merged DF Shape: " + str(df.shape))
    df = df[df["14isCustomerService"] == False]
    print("Dropped Customer Service Rows")
    print("Db Length: " + str(df.shape))
    df = df.loc[df["35firstLoginTime"] != ""]
    print("Dropped Players with no First login")
    print("Db Length: " + str(df.shape))
    df = df.loc[df["34pushSystemEnabled"] != ""]
    print("Dropped Players with no push system flag")
    print("Db Length: " + str(df.shape))
    df = df.loc[df["35firstLoginTime"] > two_months_ago]
    print("Db Length: " + str(df.shape))
    print(str(df))
    # first we need to delete the useless columns
    df = df.drop(["01date",
                  "02playerId",
                  "07accountId",
                  "08accountName",
                  "10accountDeviceId",
                  "12accountLanguage",
                  "13accountRegion",
                  "14isCustomerService",
                  "18timezoneOffsetMinutes",
                  "19ftuePhaseType",
                  "21deactivatedAtTime",
                  "23lastStateChange",
                  "24heroState",
                  "27totalPlaytime",
                  "28UAInstallTime",
                  "29UAInstallSource",
                  "30UAcpi",
                  "31UAInstallVersion",
                  "32pushDeviceId",
                  "34pushSystemEnabled",
                  "33useDayOneNotificationSettings",
                  "36maxRetention",
                  "37numberLoginDays",
                  "_id_x",
                  "_id_y",
                  "41D3",
                  "42D4",
                  "43D5",
                  "44D6",
                  "45D7",
                  "46D14",
                  "47D30",
                  "48D90",
                  "53manaConverted",
                  "60sarcophagusSeen",
                  "61sarcophagusItemsReceived"
                  ], axis=1)

    df["playerIsGuest"] = 0
    #df["playerHasDescr"] = 0
    #df["playerHasAvatar"] = 0
    df["playerLostMorePVE"] = 0
    df["totalBattles"] = 0
    df["playedMoreThanOnce"] = 0
    df["heroState"] = "Unknown"


    # !!!!! THE ALL IMPORTANT CLASSIFIER COLUMN !!!!!!
    df["churnedWithin3Days"] = 0

    print(str(list(df)))
    print(str(df))

    return df
# Delete useless columns, create new empty columns
# ------------------------------------------------------------------------------------
# This is the set of operations to be applied to each row of the dataframe.


def _format_rows(row):
    # put initial row deletion cases here
    # i.e was install Time (or first login) within last 3 months?)

    # Does the player have a player name? Are they a guest?
    player_name = str(row["03playerName"])
    if re.match("Guest[0-9]{7,}", player_name):
        row["playerIsGuest"] = 1
    else:
        row["playerIsGuest"] = 0

    # Does the player have a description?
    player_desc = str(row["04playerDescription"])
    if player_desc is not "" and player_desc is not None:
        row["04playerDescription"] = 1
    else:
        row["04playerDescription"] = 0

    # has the player set their hero avatar?
    player_avat = str(row["05heroAvatar"])
    if player_avat is not "" and player_avat is not None:
        row["05heroAvatar"] = 1
    else:
        row["05heroAvatar"] = 0

    # has the player spent money before?
    player_spender = str(row["06spender"])
    if player_spender is "False":
        row["06spender"] = 0
    else:
        row["06spender"] = 1

    # has the player given their email?
    player_email = str(row["09accountEmail"])
    if player_email is not "" and player_email is not None:
        row["09accountEmail"] = 1
    else:
        row["09accountEmail"] = 0

    # # has the player got a registered deviceId?
    # player_device = str(row["10accountDeviceId"])
    # if player_device is not "" and player_device is not None:
    #     row["10accountDeviceId"] = 1
    # else:
    #     row["10accountDeviceId"] = 0

    # player_platform = str(row["11accountPlatform"])
    # if player_platform is "":
    #     row["11accountPlatform"] = "Unknown"

    # player_language = str(row["12accountLanguage"])
    # if player_language is "":
    #     row["12accountLanguage"] = "Unknown"

    # does the player belong to an alliance
    player_alliance = str(row["15alliance"])
    if player_alliance is not "" and player_alliance is not None:
        row["15alliance"] = 1
    else:
        row["15alliance"] = 0

    # how many sessions has the user logged into?
    player_sessions = str(row["16sessionCount"])
    if player_sessions is "":
        row["16sessionCount"] = 0

    if row["16sessionCount"] > 1:
        row["playedMoreThanOnce"] = 1

    player_gold = str(row["17purchasedGoldTotal"])
    if player_gold is "":
        row["17purchasedGoldTotal"] = 0

    player_quests = str(row["20questsCompleted"])
    if player_quests is "":
        row["20questsCompleted"] = 0


    # is the player a VIP
    player_VIP = str(row["22isVip"])
    if player_VIP is not "":
        if player_VIP is "False":
            row["22isVip"] = 0
        elif player_VIP is "True":
            row["22isVip"] = 1
    else:
        row["22isVip"] = 0

    # player_hero_state = str(row["24heroState"])
    # ser = ["-2075864189.0", "1912258862.0", "58485948.0", "-948282179.0"]
    #
    # if player_hero_state in ser:
    #     if player_hero_state is "-2075864189.0":
    #         row["heroState"] = "Free"
    #     elif player_hero_state is "1912258862.0":
    #         row["heroState"] = "Captured"
    #     elif player_hero_state is "58485948.0":
    #         row["heroState"] = "Sacrificed"
    #     elif player_hero_state is "-948282179.0":
    #         row["heroState"] = "Resurrecting"

    player_messages = str(row["25messageCount"])
    if player_messages is "" or None:
        row["25messageCount"] = 0

    player_avg_session = str(row["26avgSessionLength"])
    if player_avg_session is "" or None:
        row["26avgSessionLength"] = 0

    # player_push_enabled = str(row["34pushSystemEnabled"])
    # if player_push_enabled is "False":
    #     row["34pushSystemEnabled"] = 0
    # elif player_push_enabled is "True":
    #     row["34pushSystemEnabled"] = 1

    player_day_one_login = str(row["38D0"])
    if player_day_one_login is "False":
        row["38D0"] = 0
    elif player_day_one_login is "True":
        row["38D0"] = 1

    player_day_two_login = str(row["39D1"])
    if player_day_two_login is "False":
        row["39D1"] = 0
    elif player_day_two_login is "True":
        row["39D1"] = 1

    player_day_three_login = str(row["40D2"])
    if player_day_two_login is "False":
        row["40D2"] = 0
    elif player_day_two_login is "True":
        row["40D2"] = 1

    player_castle_level = str(row["49castleLevel"])
    if player_castle_level is "":
        row["49castleLevel"] = 0

    player_hero_level = str(row["50heroLevel"])
    if player_hero_level is "":
        row["50heroLevel"] = 0

    player_power_total = str(row["51powerTotal"])
    if player_power_total is "":
        row["51powerTotal"] = 0

    player_power_hero = str(row["52powerHero"])
    if player_power_total is "":
        row["52powerHero"] = 0


    # player_mana = str(row["53manaConverted"])
    # if player_mana is "":
    #     row["53manaConverted"] = 0
    # else:
    #     row["53manaConverted"] = 1

    player_enhanced_skill = str(row["54skillEnhanced"])
    if player_enhanced_skill is "":
        row["54skillEnhanced"] = 0
    else:
        row["54skillEnhanced"] = 1

    player_constructed_ark = str(row["56arkDockConstructed"])
    if player_constructed_ark is "False":
        row["56arkDockConstructed"] = 0
    elif player_constructed_ark is "True":
        row["56arkDockConstructed"] = 1

    player_hired_commander = str(row["57commanderHired"])
    if player_hired_commander is "":
        row["57commanderHired"] = 0
    else:
        row["57commanderHired"] = 1

    player_produced_monsters = str(row["58producedMonsters"])
    if player_produced_monsters is "":
        row["58producedMonsters"] = 0
    else:
        row["58producedMonsters"] = 1

    player_researched = str(row["59researched"])
    if player_researched is "False":
        row["59researched"] = 0
    elif player_researched is "True":
        row["59researched"] = 1

    # player_seen_sarcophagus = str(row["60sarcophagusSeen"])
    # if player_seen_sarcophagus is "False":
    #     row["60sarcophagusSeen"] = 0
    # elif player_seen_sarcophagus is "True":
    #     row["60sarcophagusSeen"] = 1

    # player_collected_sarcophagus = str(row["61sarcophagusItemsReceived"])
    # if player_collected_sarcophagus is "False":
    #     row["61sarcophagusItemsReceived"] = 0
    # elif player_collected_sarcophagus is "True":
    #     row["61sarcophagusItemsReceived"] = 1

    player_subjugations = str(row["62subjugations"])
    if player_subjugations is "":
        row["62subjugations"] = 0

    player_pvp_w = str(row["63battlesWonPvP"])
    if player_pvp_w is "":
        row["63battlesWonPvP"] = 0

    player_pve_w = str(row["64battlesWonPvE"])
    if player_pve_w is "":
        row["64battlesWonPvE"] = 0

    player_deck = str(row["65deckSize"])
    if player_deck is "":
        row["65deckSize"] = 0

    player_pvp_l = str(row["66battlesLostPvP"])
    if player_pvp_l is "":
        row["66battlesLostPvP"] = 0

    player_pve_l = str(row["67battlesLostPvE"])
    if player_pve_l is "":
        row["67battlesLostPvE"] = 0


    player_PVE_won = row["64battlesWonPvE"]
    player_PVE_lost = row["67battlesLostPvE"]
    player_PVP_won = row["63battlesWonPvP"]
    player_PVP_lost = row["66battlesLostPvP"]

    if player_PVE_won < player_PVE_lost:
        row["playerLostMorePVE"] = 1

    total_battles = player_PVE_lost + player_PVE_won + player_PVP_lost + player_PVP_won

    row["totalBattles"] = total_battles

    player_first_log_date = dateparser.parse(str(row["35firstLoginTime"]))
    player_last_disconnect_date = dateparser.parse(str(row["lastDisconnect"]))

    if player_day_one_login is "True" and player_day_two_login is "True" and player_day_three_login is "True":
        row["churnedWithin3Days"] = 0
    elif player_day_one_login is "True" and player_day_two_login is "True" and player_day_three_login is "False" and player_first_log_date > (player_last_disconnect_date - rd.relativedelta(days=3)):
        row["churnedWithin3Days"] = 1
    elif player_day_one_login is "True" and player_day_two_login is "False" and player_day_three_login is "False" and player_first_log_date > (player_last_disconnect_date - rd.relativedelta(days=3)):
        row["churnedWithin3Days"] = 1

    return row

# This is the set of operations to be applied to each row of the dataframe.
# ------------------------------------------------------------------------------------
# This is the set of operations to be create dummy feature columns and also to clean columns which are no longer needed


def _format_columns(dfr):

    cols_to_transform = ["11accountPlatform", "heroState"]
    dfr = pd.get_dummies(dfr, columns=cols_to_transform)

    dfr = dfr.drop(["03playerName", "35firstLoginTime", "lastDisconnect", "38D0"], axis=1)

    return dfr


# This is the set of operations to be create dummy feature columns and also to clean columns which are no longer needed
# ------------------------------------------------------------------------------------
# Split the data into training and test sets

def get_features(dataframe):

    print(str(list(dataframe)))

    features = dataframe.drop(["churnedWithin3Days"], axis=1).values
    #why is churnedWithin3Days being included?
    print(features)
    return features


def get_classifications(dataframe):

    classifications = list(dataframe["churnedWithin3Days"].values)
    return classifications

# Split the data into training and test sets
# ------------------------------------------------------------------------------------



#def start_dataframe_prep():

dframe = _open_database_get_dataframe()

df = dframe.apply(_format_rows, broadcast=True, reduce=False, axis=1)

df = _format_columns(df)

msk = np.random.rand(len(df)) < 0.8

training_set = df[msk]
test_set = df[~msk]


print("Length of df: ", len(df))
print("Length of train: ", len(training_set))
print("Length of test: ", len(test_set))

training_set_features = get_features(training_set)
training_set_classifications = get_classifications(training_set)

test_set_features = get_features(test_set)
test_set_classifications = get_classifications(test_set)

random_forest = Classifiers.train_random_forest(training_set_features, training_set_classifications)
Classifiers.get_random_forest_accuracy(test_set_features, test_set_classifications)

columns = training_set.drop(["churnedWithin3Days"], axis=1).columns
Classifiers.get_random_forest_feature_importance(training_set_features, columns)

