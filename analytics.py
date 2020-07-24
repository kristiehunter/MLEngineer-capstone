import pandas as pd
import json

def engagement_rate(data):

    offers = {}
    offers_seen = []
    
    for _, row in data.iterrows():
        offerID = row["offer_id"]
        if row["prediction"] == 1:
            if offerID in offers_seen:
                offers[offerID]["numTotal"] += 1
                offers[offerID]["numViewed"] += int(row["offer_viewed"])
            else:
                offers_seen += [offerID]
                offers[offerID] = {
                    "numViewed": int(row["offer_viewed"]),
                    "numTotal": 1
                }
    for o in offers.keys():
        print("=== OFFER " + o + "===")
        print("Num Viewed: ", str(offers[o]["numViewed"]))
        print("Num Total: ", str(offers[o]["numTotal"]))
        ER = offers[o]["numViewed"]/offers[o]["numTotal"]
        print("Engagement Rate", str(ER))


def average_transaction_value(data):

    successful_offers = {}
    unsuccessful_offers = {}
    successful_offers_seen = []
    unsuccessful_offers_seen = []

    consumers = {}
    consumers_seen = []

    for _, row in data.iterrows():
        offerID = row["offer_id"]
        duration = row["timeline_end"] - row["timeline_start"]
        if row["prediction"] == 1:
            if offerID in successful_offers_seen:
                successful_offers[offerID]["sumTxn"] += row["transaction_window_values"]
                successful_offers[offerID]["sumTime"] += duration
            else:
                successful_offers_seen += [offerID]
                successful_offers[offerID] = {
                    "sumTxn": row["transaction_window_values"],
                    "sumTime": duration
                }

            if row["consumer_id"] in consumers_seen:
                consumers_seen += [row["consumer_id"]]
                consumers[row["consumer_id"]] += 1
            else:
                consumers[row["consumer_id"]] = 1
        else:
            if offerID in unsuccessful_offers_seen:
                unsuccessful_offers[offerID]["sumTxn"] += row["transaction_window_values"]
                unsuccessful_offers[offerID]["sumTime"] += duration
            else:
                unsuccessful_offers_seen += [offerID]
                unsuccessful_offers[offerID] = {
                    "sumTxn": row["transaction_window_values"],
                    "sumTime": duration
                }

    for o in successful_offers.keys():
        print("=== OFFER " + o + "===")
        print("Sum Txn - S: ", str(successful_offers[o]["sumTxn"]))
        print("Sum Time - S: ", str(successful_offers[o]["sumTime"]))
        if o in unsuccessful_offers.keys():
            print("Sum Txn - U: ", str(unsuccessful_offers[o]["sumTxn"]))
            print("Sum Time - U: ", str(unsuccessful_offers[o]["sumTime"]))
        else:
            print("NEVER UNSUCCESSFUL")

    return consumers

def customer_retention(data, offer_data):

    consumers = {}
    consumers_seen = []
    totalTxns = 0

    for _, row in data.iterrows():
        consumerID = row["person"]
        if consumerID in consumers_seen:
            if row["event"] == "transaction":
                consumers[consumerID]["totalTxn"] += 1
                consumers[consumerID]["sumTxn"] += row["value"]["amount"]
                totalTxns += 1
        else:
            if row["event"]== "transaction":
                consumers_seen += [consumerID]
                consumers[consumerID] = {
                    "totalTxn": 1,
                    "sumTxn": row["value"]["amount"],
                    "numOffers": 0,
                    "numSuccessful": 0
                }
                totalTxns += 1

    for _, row in offer_data.iterrows():
        consumerID = row["consumer_id"]
        if consumerID in consumers.keys():
            consumers[consumerID]["numOffers"] += 1
            consumers[consumerID]["numSuccessful"] += row["target"]
    
    print("TOTAL TRANSACTIONS: ", str(totalTxns))

    return consumers

def customer_lifetime_value(consumer_retention, consumer_data, consumer_transactions):

    consumer_lifetime = {}

    for c in consumer_retention.keys():
        i = consumer_data[consumer_data["id"] == c].index[0]
        consumer_lifetime[c] = {
            "memberSince": consumer_data.at[i, "became_member_on"],
            "consumerID": c,
            "totalTxn": consumer_retention[c]["totalTxn"],
            "sumTxn": consumer_retention[c]["sumTxn"],
            "numSuccessful": consumer_retention[c]["numSuccessful"],
            "numOffers": consumer_retention[c]["numOffers"]
        }

    with open('consumer_lv.json', 'w') as f:
        json.dump(consumer_lifetime, f)

if __name__ == "__main__":

    print("=============================\n=====   Program Start   =====\n=============================")

    # Step 1: Load data
    all_data = pd.read_csv("predict_testing.csv", sep=",")
    files = ["data/transcript.json", "data/profile.json"]
    filesObj = []
    for f in files:
        with open(f, 'r') as filename:
            print("Reading file: " + f)
            obj = json.load(filename)
            filesObj += [obj]
    
    original_data = pd.DataFrame.from_records(filesObj[0])
    consumer_data = pd.DataFrame.from_records(filesObj[1])
    print("----- Predicted Data -----")
    print(all_data.head())

    # Step 2: Calculate Engagement Rate for Successful Rewards
    engagement_rate(all_data)

    # Step 3: Calculate Average Transaction Value
    consumer_transactions = average_transaction_value(all_data)

    # # Step 4: Calculate Customer Retention
    consumer_retention = customer_retention(original_data, all_data)

    # # Step #5: Calculate Customer Lifetime Value
    customer_lifetime_value(consumer_retention, consumer_data, consumer_transactions)
