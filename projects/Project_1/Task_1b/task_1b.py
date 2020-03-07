#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project1, task b, which endeavours to perform linear regression with
feature transformation

Example usage from CLI:
 $ python3 task_1b.py --train ~/path/to/train/dir/ --weights ~/path/to/weights/file

For help, run:
 $ python3 task_1b.py -h

To do:
- Improve score to meet hard baseline from project
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import argparse
import os

__author__ = "Josephine Yates; Philip Hartout; Flavio Rump"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch; flrump@student.ethz.ch"

def rmse(y_true,y_pred):
    """This function computes the RMSE of a vector of predicted and true labels

    Args:
        y_true (numpy.ndarray): Vector of true labels.
        y_pred (numpy.ndarray): Vector of predicted labels.

    Returns:
        float: the computed root mean square error
    """
    return mean_squared_error(y_true, y_pred) ** 0.5


def feature_transformation(X):
    """This function calculates new df with transformed features - linear,
    quadratic, exponential, cosine and constant - starting with
    X = [x1,x2,x3,x4,x5]

    Args:
        X (pandas.core.frame.DataFrame): Matrix containing the training data.

    Returns:
        pandas.core.frame.DataFrame: the transformed features
    """
    # apply function to df
    qX = X.apply(np.square)
    # rename columns
    qX.columns = ["phi6","phi7","phi8","phi9","phi10"]

    eX = X.apply(np.exp)
    eX.columns = ["phi11","phi12","phi13","phi14","phi15"]

    cX = X.apply(np.cos)
    cX.columns = ["phi16","phi17","phi18","phi19","phi20"]

    cst = pd.DataFrame(np.ones((X.shape[0],)),columns=["phi21"])

    X.columns = ["phi1","phi2","phi3","phi4","phi5"]

    return pd.concat([X,qX,eX,cX,cst],axis=1)


def evaluate_regression(X_train,y_train):
    """This function computes the average RMSE based on k-fold cross-validation

    Args:
        y_train (numpy.ndarray): Matrix of training samples.
        y_train (numpy.ndarray): Vector of training labels.

    Returns:
        float: the average root mean square error
    """
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_avg=0
    for train_index, test_index in kf.split(X_train):
        # separate for cross validation
        X_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_cv, y_test_cv = y_train[train_index], y_train[test_index]
        # fit the model
        model = LinearRegression().fit(X_cv,y_cv)
        # predict
        y_pred = model.predict(X_test_cv)
        # calculate average RMSE
        rmse_avg +=rmse(y_test_cv,y_pred)
    return rmse_avg/n_splits


def main():
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    X_train, y_train = df_train.drop(['y'], axis=1),df_train['y'].values

    # Scale the data
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = pd.DataFrame(scaler.fit_transform(X_train))

    # Create the dataset with transformed features
    X_train = feature_transformation(X_train).values

    # Train the model and get models
    model = LinearRegression().fit(X_train,y_train)
    weights = model.coef_
    reg_score = model.score(X_train,y_train)
    print(f"The reg score is {reg_score}")
    score = evaluate_regression(X_train, y_train)
    print(f"Average RMSE on 10-fold cv is {score}")

    # Export to .csv
    weight_df = pd.DataFrame(weights)
    weight_df.to_csv(FLAGS.weights, sep=" ", index=False, header=False, float_format='%.2f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="path where the CSV file where weights should be written")
    FLAGS = parser.parse_args()
    main()
