#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project1, task b, which endeavours to perform linear regression with 
feature transformation

Example usage from CLI:
 $ python3 task_1b.py --train ~/path/to/train/dir/ --w ~/path/to/weights/file 

For help, run:
 $ python3 task_1b.py -h
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

def RMSE(y_true,y_pred):
    # calculate root mean squared error
    return mean_squared_error(y_true, y_pred)**0.5

def feature_transformation(X):
    # calculate new df with transformed features - linear, quadratic,
    # exponential, cosine and constant - starting with X = [x1,x2,x3,x4,x5]
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
    kf = KFold(n_splits=10)
    RMSE_avg=0
    for train_index, test_index in kf.split(X_train):
        # separate for cross validation
        X_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_cv, y_test_cv = y_train[train_index], y_train[test_index]
        # fit the model
        model = LinearRegression().fit(X_cv,y_cv)
        # predict
        y_pred = model.predict(X_test_cv)
        # calculate average RMSE
        RMSE_avg +=RMSE(y_test_cv,y_pred)/10
    return RMSE_avg

def main():
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    X_train, y_train = df_train.drop(['y'], axis=1),df_train['y'].values
    # create the dataset with transformed features
    X_train = feature_transformation(X_train).values

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # train the model and get models
    model = LinearRegression().fit(X_train,y_train)
    weights = model.coef_
    reg_score = model.score(X_train,y_train)
    print("The reg score is ",reg_score)
    score = evaluate_regression(X_train, y_train)
    print("Average RMSE on 10-fold cv is ",score)

    # export to .csv
    weight_df = pd.DataFrame(weights)
    weight_df.to_csv(FLAGS.w, sep=" ", index=False, header=False, float_format='%.2f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--w", type=str, required=True,
                        help="path where the CSV file where weights should be written")
    FLAGS = parser.parse_args()
    main()
