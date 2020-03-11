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

__author__ = "Josephine Yates; Philip Hartout; Flavio Rump"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch; flrump@student.ethz.ch"


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys

sys.settrace

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


def main():
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    x_train, y_train = df_train.drop(['y'], axis=1), df_train['y'].values

    # Create the dataset with transformed features
    x_train = feature_transformation(x_train).values

    # Scale the data
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train = pd.DataFrame(scaler.fit_transform(x_train)).values

    # Actual modelling
    clf = LassoCV(eps=0.001, n_alphas=100, alphas=np.arange(10000,100000,1),\
                   fit_intercept=True, normalize=False, precompute='auto', \
                   max_iter=1000, tol=0.0001, copy_X=False, cv=10, \
                   verbose=2, n_jobs=1, positive=False, random_state=42,\
                   selection='cyclic').fit(x_train, y_train)

    print(f"Alpha chosen: {clf.alpha_}")

    # export to .csv
    weight_df = pd.DataFrame(clf.coef_)
    weight_df.to_csv(FLAGS.weights, sep=" ", index=False, header=False, float_format='%.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file \
    directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training \
                        data")
    parser.add_argument("--weights", "-w", type=str, required=True,
                        help="path where the CSV file where weights should be \
                        written")
    FLAGS = parser.parse_args()
    main()
