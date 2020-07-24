import pandas as pd
import numpy as np
import math
import json

def encode_column(dataframe, column, values):
    """ 
        For any string valued columns, encode each unique value 
        with an integer value.
    """

    value_nums = {}

    start = 0
    for v in values:
        value_nums[str(v)] = start
        start += 1

    for index, _ in dataframe.iterrows():
        current_value = dataframe.iloc[index][column]
        dataframe.at[index, column] = value_nums[str(current_value)]

    return dataframe


def transfer_data(transcript, transcript_merged):
    """
    Copy the data from the original transcript data object 
    into a readable/parseable version of merged data.  For 
    each person, aggregate all of their transactions in an 
    array.
    """

    test_counter = 0

    # For each row in the transcript, extract data
    for index, row in transcript.iterrows():
        # if test_counter == 50000:
        #     break
        obj = {
            "event": row["event"],
            "time": row["time"],
            "value": row["value"]
        }
        # If the person alread exists in the merged data, append the transaction 
        # to that row, otherwise append a new row to the new dataframe
        if row["person"] not in transcript_merged:
            new_data = pd.DataFrame({'person': [row["person"]], 'transcript_objects': [obj]}, index=[index])
            transcript_merged = pd.concat([transcript_merged, new_data], axis=0)
        else:
            transcript_merged.loc[index, "transcript_objects"] += obj
        print("Transaction data counter: ", str(test_counter))
        test_counter += 1

    return transcript_merged


def extract_files(files):
    """
    Read all JSON files into an array.
    """
    filesObj = []

    for f in files:
        with open(f, 'r') as filename:
            print("Reading file: " + f)
            obj = json.load(filename)
            filesObj += [obj]

    return filesObj


def extract_ids(data):
    """
        Return array of IDs from data frame.
    """
    id_list = []

    for _, row in data.iterrows():
        id_list += [row["id"]]

    return id_list

## TODO: Remove if this wors in the encoder
def get_offer_type_enum(offer_type):
    """

    """

    if offer_type == "bogo":
        return 0
    elif offer_type == "informational":
        return 1
    elif offer_type == "discount":
        return 2
    else:
        return 3


def get_channel_enum(channel_array, channel_type):
    """
        Return true if the offer was presented in the
        channel type given.
    """
    return channel_type in channel_array



def add_to_base_dataframe(merged_data, profile, offer):
    """
        Append a row of new data to the dataframe.
    """

    index_counter = merged_data.shape[0]

    new_data = pd.DataFrame({
        "consumer_gender": profile["gender"],
        "consumer_member_start": profile["became_member_on"],
        "consumer_age": profile["age"],
        "consumer_income": profile["income"],
        "consumer_id": profile["id"],
        "offer_id": offer["id"],
        "offer_type": offer["offer_type"],
        "offer_duration": offer["duration"],
        "offer_difficulty": offer["difficulty"],
        "offer_reward": offer["reward"],
        "offer_web": get_channel_enum(offer["channels"], "web"),
        "offer_email": get_channel_enum(offer["channels"], "e-mail"),
        "offer_mobile": get_channel_enum(offer["channels"], "mobile"),
        "offer_social": get_channel_enum(offer["channels"], "social"),
        "offer_received": False,
        "offer_viewed": False,
        "offer_completed": False,
        "transaction_window_values": 0,
        "timeline_start": 0,
        "timeline_end": 0,
        "repeat_offer": False
    }, index=[index_counter])
    merged_data = pd.concat([merged_data, new_data], axis=0)

    return merged_data


def create_consumer_dataframe(profile_data):
    """
        Creates the base dataframe for consumer transaction
        data for analytics post-prediction.
    """
    consumers = pd.DataFrame()
    index = 0

    for _, consumer in profile_data.iterrows():
        new_data = pd.DataFrame({
            "consumer_id": consumer["id"],
            "transaction_count": 0,
            "cumulative_transactions": 0,
            "average_transactions": 0
        }, index=[index])
        index += 1
        consumers = pd.concat([consumers, new_data], axis=0)

    return consumers



def populate_consumer_data(merged_data, offers, consumers, transactions, consumer_transactions):
    """
        For each transaction, update the dataframes with the consumer,
        offer and transaction data.
    """

    # For each consumer, get the correlated transactions 
    # from transcript.json
    for _, row in consumers.iterrows():
        txn_dict = get_transaction_data(row["id"], transactions)
        # For every offer mentioned in those transactions
        # Add each offer related value to the dataframe and 
        # aggregate the non-offer transactions
        for offer_id in txn_dict.keys():
            if offer_id != "transaction":
                offer_row = offers[offers["id"] == offer_id].squeeze()
                merged_data = add_to_base_dataframe(merged_data, row, offer_row)
                merged_data = add_offer_transaction_data(merged_data, offer_row, txn_dict, row)
        if "transaction" in txn_dict.keys():
            merged_data, consumer_transactions = update_transaction_values(merged_data, consumer_transactions,
                txn_dict["transaction"], row["id"], offers)

    return merged_data


def update_transaction_values(merged_data, consumer_transactions, transactions, consumer_id, offers):
    """
        For all transactions with monetary value, determine whether
        or not it falls within the offer window (between when the offer
        was received and when it was either completed or expired).  Any
        offer falling outside that window is counted as an extraneous
        transaction for analytics.
    """

    # Check that the consumer is in the existing data frame
    if consumer_id in merged_data["consumer_id"].values:

        # Get all data in the data frame related to that consumer
        data_subset = merged_data[merged_data["consumer_id"] == consumer_id]

        for t in transactions:
            found_matching = False

            # For all existing data in the dataframe, compare the current 
            # transaction with it to determine whether or not it's related 
            # to that offer
            for i, row in data_subset.iterrows():
                txn_time = t["time"]
                offer_start = row["timeline_start"]
                offer_end = row["timeline_end"]

                # If the current offer does NOT have an ending (ie the 
                # transaction was completed), set the end time as the 
                # expiry date
                if offer_end == 0:
                    # NOTE: The expiry date from the portfolio is set in days
                    # and the timestamps are set in hours
                    offer_end = int(offer_start + (offers[offers["id"] == row["offer_id"]]["duration"].values[0] * 24))
                    merged_data.at[i, "timeline_end"] = offer_end

                # If the transaction falls in the offer window, 
                # add it to the value contributing towards this offer
                if txn_time <= offer_end and txn_time > offer_start:
                    merged_data.at[i, "transaction_window_values"] += t["value"]["amount"]
                    found_matching = True

            # If no offer was found for this transaction, add it to the
            # consumer transaction dataframe
            if not found_matching:
                j = consumer_transactions[consumer_transactions["consumer_id"] == consumer_id].index[0]
                consumer_transactions.at[j, "transaction_count"] += 1
                consumer_transactions.at[j, "cumulative_transactions"] += t["value"]["amount"]

    # If consumer was not in the dataframe, they did not received any offers
    # and the data should be added to teh consumer transaction dataframe
    else:
        for t in transactions:
            j = consumer_transactions[consumer_transactions["consumer_id"] == consumer_id].index[0]
            consumer_transactions.at[j, "transaction_count"] += 1
            consumer_transactions.at[j, "cumulative_transactions"] += t["value"]["amount"]

    return merged_data, consumer_transactions


def add_offer_transaction_data(merged_data, offer, txn_dict, consumer):
    """
        For a transaction relating to an offer, update the dataframe
        with the event.  Note that some offers are received more than
        once and therefore updated the events will need to take into
        account which instance of the offer belongs to which event.
    """

    offer_counter = 0
    consumer_id = consumer["id"]
    offer_id = offer["id"]
    for t in txn_dict[offer_id]:
        if consumer_id == "e12aeaf2d47d42479ea1c4ac3d8286c6":
            print(t)
        if t["event"] == "offer received":
            # If the offer has not been seen before, it is the first one
            # in the dataframe
            if offer_counter == 0:
                merged_data = update_offer_received(offer_counter, merged_data, consumer_id, offer_id, t["time"],
                    consumer, offer, False)
                offer_counter += 1
            # If counter is > 0, the transaction should be marked as a 
            # repeat offer in the dataframe
            else:
                merged_data = update_offer_received(offer_counter, merged_data, consumer_id, offer_id, t["time"],
                    consumer, offer, True)
                offer_counter += 1
        elif t["event"] == "offer completed":
            j = offer_counter - 1
            merged_data = update_offer_completed(j, merged_data, consumer_id, offer_id, t["time"])
        else:
            j = offer_counter - 1
            merged_data = update_offer_viewed(j, merged_data, consumer_id, offer_id, t["time"])
        
    return merged_data


def update_offer_received(offer_counter, merged_data, consumer_id, offer_id, offer_start, consumer, offer, repeat):
    """
        Update the dataframe with information related to an offer
        being received.  If the offer is a repeat, it needs to be 
        added a second time to the dataframe and marked as a repeat.
    """

    if repeat:
        merged_data = add_to_base_dataframe(merged_data, consumer, offer)
    
    i = get_row_indices(merged_data, consumer_id, offer_id)
    merged_data.at[i[offer_counter], "offer_received"] = True
    merged_data.at[i[offer_counter], "timeline_start"] = offer_start

    if repeat:
        merged_data.at[i[offer_counter], "repeat_offer"] = True

    return merged_data
        
        
def update_offer_completed(offer_counter, merged_data, consumer_id, offer_id, offer_end):
    """
        Update the dataframe with information related to an offer
        being completed.
    """
    i = get_row_indices(merged_data, consumer_id, offer_id)
    merged_data.at[i[offer_counter], "offer_completed"] = True
    merged_data.at[i[offer_counter], "timeline_end"] = offer_end

    return merged_data

def update_offer_viewed(offer_counter, merged_data, consumer_id, offer_id, offer_end):
    """
        Update the dataframe with information related to an offer
        being viewed.
    """
    i = get_row_indices(merged_data, consumer_id, offer_id)
    merged_data.at[i[offer_counter], "offer_viewed"] = True

    return merged_data


def get_row_indices(merged_data, consumer_id, offer_id):
    """
        Return the row indices related to a specific consumer and a
        specific offer.
    """

    row = merged_data[(merged_data["offer_id"] == offer_id) & (merged_data["consumer_id"] == consumer_id)]

    return row.index


def get_transaction_data(consumer_id, transactions):
    """
        For all transactions related to a consumer, return 
        a dictionary with the keys being the offer IDs or the
        value 'transaction' for non-offer related events.
    """

    txn_dict = {}

    txn = transactions[transactions["person"] == consumer_id]
    for obj in txn["transcript_objects"]:
        if "offer id" in obj["value"].keys():
            offer_id = obj["value"]["offer id"]
        elif "offer_id" in obj["value"].keys():
            offer_id = obj["value"]["offer_id"]
        else:
            offer_id = "transaction"
        
        if offer_id not in txn_dict.keys():
            txn_dict[offer_id] = [obj]
        else:
            txn_dict[offer_id] += [obj]
    
    return txn_dict


if __name__ == "__main__":
    
    # Step 1: Extract Data
    files = ["data/portfolio.json", "data/profile.json", "data/transcript.json"]
    filesObj = extract_files(files)

    # Data frames of JSON files
    portfolio = pd.DataFrame.from_records(filesObj[0])
    profile = pd.DataFrame.from_records(filesObj[1])
    transcript = pd.DataFrame.from_records(filesObj[2])

    # Remove any outlier data in the consumer profiles
    profile_clean = profile.dropna()
    consumer_ids = extract_ids(profile_clean)
    print("===== Profile Data Sample =====")
    print(profile_clean.head(5))
    print("Number of consumers: ", str(len(consumer_ids)))

    # Extract Offer IDs from portfolio to build timelines
    offer_ids = extract_ids(portfolio)
    print("===== Portfolio Data Sample =====")
    print(portfolio.head(5))
    print("Number of offers: ", str(len(offer_ids)))

    print("===== Transcript Data Sample =====")
    print(transcript.head(5))

    # Step 2: Create a base dataframe for each person to be paired with the offer data
    base_dataframe = pd.DataFrame(columns=["consumer_gender", "consumer_member_start",
        "consumer_age", "consumer_income", "consumer_id", "offer_id", "offer_type", "offer_duration",
        "offer_difficulty", "offer_reward", "offer_web", "offer_email", "offer_mobile",
        "offer_social", "offer_received", "offer_completed", "transaction_window_values",
        "timeline_start", "timeline_end", "repeat_offer"])

    # Step 3: Create a vase dataframe for each person that will 
    # track their transactions outside of offer windows 
    consumer_transactions = create_consumer_dataframe(profile_clean)

    # Step 4: Combine all transcactions by person (consumer_id) from the transcript file
    transcript_new = pd.DataFrame()
    transcript_merged = transfer_data(transcript, transcript_new)
    print("===== Transcript Merged Data Sample =====")
    print(transcript_merged.head(5))

    # Step 5: For the base dataframe, populate the empty columns with the transaction data
    updated_data = populate_consumer_data(base_dataframe, portfolio, profile_clean, transcript_merged,
        consumer_transactions)
    print("===== Full Data Sample =====")
    print(updated_data.head(5))

    # Step 6: Encode non-numeric values
    updated_data = encode_column(updated_data, "consumer_gender", ["M", "F", "O"])
    updated_data = encode_column(updated_data, "offer_type", ["bogo", "informational", "discount"])
    updated_data[['offer_received', 'offer_completed', 'offer_web', 'offer_email', 'offer_mobile', 'offer_social', 'repeat_offer']] = updated_data[['offer_received', 'offer_completed', 'offer_web', 'offer_email', 'offer_mobile', 'offer_social', 'repeat_offer']].astype(int)
    print("===== Encoded Data Sample =====")
    print(updated_data.head(5))

    updated_data.to_csv('results.csv', index=False, header=True)
    consumer_transactions.to_csv('consumers.csv', index=False, header=True)

