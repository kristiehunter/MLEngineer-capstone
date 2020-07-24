from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
import pandas as pd
import copy

class StarbucksData:

    def __init__(self):
        self.X = None
        self.Y = None
        self.trainX = None
        self.trainY = None
        self.validateX = None
        self.validateY = None
        self.testX = None
        self.testY = None
        self.metrics = {
            "training": {},
            "validation": {},
            "testing": {}
        }
        self.predictions = []

    def splitData(self):

        trainX, trainY, vtX, vtY = self.splitSection(0.4, self.X, self.Y)
        validX, validY, testX, testY = self.splitSection(0.5, vtX, vtY)

        # Create targets
        self.trainY = trainY
        self.validateY = validY
        self.testY = testY

        # Create backup copies with all the columns
        self.trainX_backup = copy.deepcopy(trainX)
        self.validateX_backup = copy.deepcopy(validX)
        self.testX_backup = copy.deepcopy(testX)
        

        # Create algorithm copies with select features
        self.trainX = copy.deepcopy(trainX.drop(columns=["consumer_id", "offer_id", "offer_received", "repeat_offer", "offer_duration",
            "timeline_start", "timeline_end", "offer_viewed", "transaction_window_values"]))
        self.validateX = copy.deepcopy(validX.drop(columns=["consumer_id", "offer_id", "offer_received", "repeat_offer", "offer_duration",
            "timeline_start", "timeline_end", "offer_viewed", "transaction_window_values"]))
        self.testX = copy.deepcopy(testX.drop(columns=["consumer_id", "offer_id", "offer_received", "repeat_offer", "offer_duration",
            "timeline_start", "timeline_end", "offer_viewed", "transaction_window_values"]))


    def splitSection(self, size, X, Y):
        splitter = StratifiedShuffleSplit(n_splits = 1, test_size = size)
        data = []
        for set1, set2 in splitter.split(X, Y):
            data += [X.iloc[set1]]
            data += [Y.iloc[set1]]
            data += [X.iloc[set2]]
            data += [Y.iloc[set2]]
        
        
        return data[0], data[1], data[2], data[3]
    
    def calculateClassCount(self, dataType):

        counts = None

        if dataType == "train":
            counts = self.trainY.value_counts()
        elif dataType == "validate":
            counts = self.validateY.value_counts()
        else:
            counts = self.testY.value_counts()
        
        return counts