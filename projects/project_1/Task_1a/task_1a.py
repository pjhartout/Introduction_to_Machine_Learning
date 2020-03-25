#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project1, task a, which uses a 10-fold cross validation to evaluate
ridge regression with different regularization parameters.
Example usage from CLI:
 $ python3 task_1a.py --train ~/path/to/train/dir/ --RMSE ~/path/to/RMSE/file
For help, run:
 $ python3 task_1a.py -h
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
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


def main():
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    x_train, y_train = df_train.drop(['y'], axis=1).values, df_train['y'].values

    # regularization parameters
    alphas = [0.01, 0.1, 1, 10, 100]


    # cross validation
    rmse_list = [0] * len(alphas)
    for i, alpha in enumerate(alphas):
        # cross validation iterator
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(x_train):
            x_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_cv, y_test_cv = y_train[train_index], y_train[test_index]
            # fit the model
            model = Ridge(alpha=alpha).fit(x_cv, y_cv)
            y_pred = model.predict(x_test_cv)
            # average RMSE computation
            rmse_list[i] += rmse(y_test_cv, y_pred) / 10
    # export to .csv
    rmse_df = pd.DataFrame(rmse_list)
    rmse_df.to_csv(FLAGS.score, sep=" ", index=False, header=False, float_format='%.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--score", type=str, required=True,
                        help="path where the CSV file for score should be written")
    FLAGS = parser.parse_args()
    main()