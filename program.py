import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn import preprocessing
import math
import json
import time

from StarbucksData import StarbucksData


# Logistic Regression
def method_logistic_regression(X, Y):
    model = LogisticRegression()
    parameters = {
        "penalty": ["l1"],
        "C": [0.1, 0.15, 0.2],
        "fit_intercept": [True, False],
        "solver": ["liblinear"],
        "max_iter": [500, 650, 800]
    }

    grid_result = GridSearchCV(model, parameters, cv=10)
    grid_result.fit(X, Y)
    
    return grid_result


# Decision Tree Classifier
def method_tree_classifier(X, Y):
    model = DecisionTreeClassifier()
    parameters = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [5, 8, 10],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [1, 2, 3],
        "max_features": [None]
    }

    grid_result = GridSearchCV(model, parameters, cv=10)
    grid_result.fit(X, Y)

    return grid_result


# Multilayer Perceptron Classifier
def method_neural_network(X, Y):
    model = MLPClassifier()
    parameters = {
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "invscaling", "adaptive"]
    }
    grid_result = GridSearchCV(model, parameters, cv=10)
    grid_result.fit(X, Y)

    return grid_result


# Gaussian Naive Bayes Classifier
def method_bayes_classifier(X, Y):
    model = GaussianNB()
    parameters = {}
    
    grid_result = GridSearchCV(model, parameters, cv=10)
    grid_result.fit(X, Y)

    return grid_result


# K-Nearest Neighbours Classifier
def method_knn(X, Y):
    model = KNeighborsClassifier()
    parameters = {
        "n_neighbors": [8, 10, 12, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree"],
        "p": [1],
        "leaf_size": [3, 5, 80]
    }
    
    grid_result = GridSearchCV(model, parameters, cv=10)
    grid_result.fit(X, Y)

    return grid_result


def calculate_metrics(Y_estimate, Y):
    """
        Given the actual target value and the prediction,
        calculate the Matthews Correlation 
        Coefficient and the Accuracy.
    """

    matthews = matthews_corrcoef(Y, Y_estimate)
    accuracy = accuracy_score(Y, Y_estimate)

    metrics = {  "matthews": matthews, "accuracy": accuracy }
    print(metrics)

    return metrics


def predict_data(SD, predict_type, models):
    """
        For all trained models, predict the target given the data
        and calculate the metrics.  Update the StarbucksData class 
        with the metrics and export the predictions for analytics 
        to a csv.
    """

    X = None
    Y = None
    filename = None
    prediction_dataframe = None

    if predict_type == "validation":
        X = SD.validateX
        Y = SD.validateY
        prediction_dataframe = SD.validateX_backup.copy()
        filename = "predict_validate.csv"
    elif predict_type == "testing":
        X = SD.testX
        Y = SD.testY
        prediction_dataframe = SD.testX_backup.copy()
        filename = "predict_testing.csv"
    else: 
        X = SD.trainX
        Y = SD.trainY
        prediction_dataframe = SD.trainX_backup.copy()
        filename = "predict_training-DT.csv"

    best_score = 0

    for m in models.keys():
        print("Looking at model " + m + " for predicting data of type " + predict_type)
        estimator = models[m]["estimator"]
        prediction = estimator.predict(X)
        SD.metrics[predict_type][m] = calculate_metrics(prediction, Y)

        if SD.metrics[predict_type][m]["accuracy"] > best_score:
            prediction_dataframe["target"] = Y
            prediction_dataframe["prediction"] = prediction
        
    prediction_dataframe.to_csv(filename, index=False, header=True)
    
    return models


if __name__ == "__main__":
    print("=============================\n=====   Program Start   =====\n=============================")

    # Step 1: Load data
    filename = "results_backup2.csv"
    all_data = pd.read_csv(filename, sep=",")

    # Step 2: Split features from target
    Y = all_data["offer_completed"]
    X = all_data.drop(columns=["offer_completed"])

    SD = StarbucksData()
    SD.X = X
    SD.Y = Y

    # Step 3: Split data into testing, training and validation 
    # and ensure even class ratios in the 3 sets
    SD.splitData()

    print("----- Class Distributions in Data Splits -----")
    print("Training:\n", SD.calculateClassCount("train"))
    print("Validation:\n", SD.calculateClassCount("validate"))
    print("Testing:\n", SD.calculateClassCount("test"))
    print("\n\n")
    
    # Step 4: Train all models using the training set
    best_models = {}

    ### MODEL 1: LOGISTIC REGRESSION
    print("----- Training the Models -----")
    # print("==== Calculating Logistic Regression ====")
    # start1 = time.time()
    # model1 = method_logistic_regression(SD.trainX, SD.trainY)
    # end1 = time.time()
    # best_models["logistic_regression"] = {
    #     "estimator": model1.best_estimator_,
    #     "score": model1.best_score_,
    #     "params": model1.best_params_
    # }
    # print("Best Score: ", model1.best_score_)
    # print("Training Time: ", (end1 - start1))
    # print("\n")

    ### MODEL 2: DECISION TREE
    print("==== Calculating Tree Classifier ====")
    start2 = time.time()
    model2 = method_tree_classifier(SD.trainX, SD.trainY)
    end2 = time.time()
    best_models["tree_classifier"] = {
        "estimator": model2.best_estimator_,
        "score": model2.best_score_,
        "params": model2.best_params_
    }
    print("Best Score: ", model2.best_score_)
    print("Training Time: ", (end2 - start2))
    print("\n")

    ### MODEL 3: NEURAL NETWORK
    # print("==== Calculating Neural Network ====")
    # start3 = time.time()
    # model3 = method_neural_network(SD.trainX, SD.trainY)
    # end3 = time.time()
    # best_models["neural_network"] = {
    #     "estimator": model3.best_estimator_,
    #     "score": model3.best_score_,
    #     "params": model3.best_params_
    # }
    # print("Best Score: ", model3.best_score_)
    # print("Training Time: ", (end3 - start3))
    # print("\n")

    ### MODEL 4: NAIVE BAYES
    # print("==== Calculating Bayes Classifier ====")
    # start4 = time.time()
    # model4 = method_bayes_classifier(SD.trainX, SD.trainY)
    # end4 = time.time()
    # best_models["bayes_classifier"] = {
    #     "estimator": model4.best_estimator_,
    #     "score": model4.best_score_,
    #     "params": model4.best_params_
    # }
    # print("Best Score: ", model4.best_score_)
    # print("Training Time: ", (end4 - start4))
    # print("\n")

    ### MODEL 5: KNN
    print("==== Calculating KNN ====")
    start5 = time.time()
    model5 = method_knn(SD.trainX, SD.trainY)
    end5 = time.time()
    best_models["knn"] = {
        "estimator": model5.best_estimator_,
        "score": model5.best_score_,
        "params": model5.best_params_
    }
    print("Best Score: ", model5.best_score_)
    print("Training Time: ", (end5 - start5))

    params = {}
    for m in best_models.keys():
        params[m] = best_models[m]["params"]

    with open('best_models.json', 'w') as f:
        json.dump(params, f)

    predict_data(SD, "training", best_models)
    predict_data(SD, "validation", best_models)
    predict_data(SD, "testing", best_models)

    with open('results_data.json', 'w') as f:
        json.dump(SD.metrics, f)

    