#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project1, task b, which endeavours to perform linear regression with
feature transformation

example usage from CLI:
 $ python3 task_1b.py --train ~/path/to/train/dir/ --weights ~/path/to/weights/file

For help, run:
 $ python3 task_1b.py -h

To do:
- Improve score to meet hard baseline from project
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

__author__ = "Josephine Yates; Philip Hartout; Flavio Rump"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch; flrump@student.ethz.ch"


def rmse(y_true, y_pred):
    """This function computes the RMSE of a vector of predicted and true labels

    Args:
        y_true (numpy.ndarray): Vector of true labels.
        y_pred (numpy.ndarray): Vector of predicted labels.

    Returns:
        float: the computed root mean square error
    """
    return mean_squared_error(y_true, y_pred) ** 0.5


def feature_transformation(x):
    """This function calculates new df with transformed features - linear,
    quadratic, exponential, cosine and constant - starting with
    X = [x1,x2,x3,x4,x5]

    Args:
        x (pandas.core.frame.DataFrame): Matrix containing the training data.

    Returns:
        pandas.core.frame.DataFrame: the transformed features
    """
    # apply function to df
    qx = x.apply(np.square)
    # rename columns
    qx.columns = ["phi6", "phi7", "phi8", "phi9", "phi10"]

    ex = x.apply(np.exp)
    ex.columns = ["phi11", "phi12", "phi13", "phi14", "phi15"]

    cx = x.apply(np.cos)
    cx.columns = ["phi16", "phi17", "phi18", "phi19", "phi20"]

    cst = pd.DataFrame(np.ones((x.shape[0],)), columns=["phi21"])

    x.columns = ["phi1", "phi2", "phi3", "phi4", "phi5"]

    return pd.concat([x, qx, ex, cx, cst], axis=1)


def evaluate_model(x_train, y_train, alpha, model_train):
    """This function computes the average RMSE based on k-fold cross-validation

    Args:
        x_train (numpy.ndarray): Matrix of training samples.
        y_train (numpy.ndarray): Vector of training labels.

    Returns:
        float: the average root mean square error
    """
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_avg=0
    for train_index, test_index in kf.split(x_train):
        # separate for cross validation
        x_cv, x_test_cv = x_train[train_index], x_train[test_index]
        y_cv, y_test_cv = y_train[train_index], y_train[test_index]
        # fit the model
        if model_train=="ridge":
            model = Ridge(alpha=alpha).fit(x_cv,y_cv)
        elif model_train=="lasso":
            model = Lasso(alpha=alpha).fit(x_cv,y_cv)
        # predict
        y_pred = model.predict(x_test_cv)
        # calculate average RMSE
        rmse_avg += rmse(y_test_cv, y_pred)
    return rmse_avg/n_splits

def plot_error_model(alphas,mean_errors):
    """This function plots the mean error of CV vs the regularization parameter in ridge regression

    Args:
        alphas (numpy.ndarray): regularization parameters tried out
        mean_errors (numpy.ndarray): result of average error over cross validation for different alphas

    Returns:
        None
    """
    plt.plot(alphas,mean_errors)
    plt.title("Mean cv error vs regularization parameter")
    plt.xlabel("Alpha, regularization parameter")
    plt.ylabel("Mean error")
    plt.show()


def main(method="ridge"):
    """Fitting a linear regression to the data

    Args:
        method (string): the type of LR to use: can be "LR" for classical linear regression, "ridge" for ridge 
        regression with RMSE as evaluation metric using 10-fold CV, or "ridgeCV" for ridge regression using 
        LS as evaluation metric with LOOCV. 

    Returns:
        None
    """
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    x_train, y_train = df_train.drop(['y'], axis=1), df_train['y'].values

    # Scale the data
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train = pd.DataFrame(scaler.fit_transform(x_train))

    # Create the dataset with transformed features
    x_train = feature_transformation(x_train).values

    # Train the model and get models
    if method=="LR":
        print("Starting regression with classical linear regression...")
        model = LinearRegression().fit(x_train, y_train)
        weights = model.coef_
        reg_score = model.score(x_train, y_train)
        print(f"The reg score is {reg_score}")
        score = evaluate_regression(x_train, y_train)
        print(f"Average RMSE on 10-fold cv is {score}")

    elif method=="ridge":
        print("Starting regression with ridge regression...")
        alphas = np.arange(12700,12750,0.1)
        rmse=[]
        for alpha in alphas:
            rmse.append(evaluate_model(x_train,y_train,alpha,"ridge"))
        plot_error_model(alphas,rmse)
        alpha = alphas[np.argmin(rmse)]
        print("Alpha is {}".format(alpha))
        model = Ridge(alpha=alpha).fit(x_train,y_train)
        weights = model.coef_

    elif method=="ridgeCV":
        print("Starting regression with ridge LOOCV...")
        alphas = np.arange(1.0,15,0.1)
        model = RidgeCV(alphas=alphas,fit_intercept=True,normalize=True,scoring=None,
                            cv=None,gcv_mode=None,store_cv_values=True).fit(x_train, y_train)
        errors = model.cv_values_
        print(errors.shape)
        mean_errors = [np.mean(errors[:,i]) for i in range(errors.shape[1])]
        alpha = model.alpha_
        weights = model.coef_
        plot_error_model(alphas,mean_errors)
        print("The estimated alpha is {}".format(alpha))
    
    elif method=="lasso":
        print("Starting regression with lasso regression...")
        alphas = np.arange(0.01,5,0.01)
        rmse=[]
        for alpha in alphas:
            rmse.append(evaluate_model(x_train,y_train,alpha,"lasso"))
        plot_error_model(alphas,rmse)
        alpha = alphas[np.argmin(rmse)]
        print("Alpha is {}".format(alpha))
        model = Lasso(alpha=alpha).fit(x_train,y_train)
        weights = model.coef_

    # export to .csv
    weight_df = pd.DataFrame(weights)
    weight_df.to_csv(FLAGS.weights, sep=" ", index=False, header=False, float_format='%.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="path where the CSV file where weights should be written")
    parser.add_argument("--method", "-m", type=str, required=True,
                        help="method used for optimization, can be LR (classical Linear Regression), ridge (ridge regr with CV with RMSE), ridgeCV (ridge regr with LOOCV with LS) or lasso (lasso regr with CV with RMSE)")
    FLAGS = parser.parse_args()
    main(FLAGS.method)
