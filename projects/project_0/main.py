#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project0 which predicts the class labels of a test set based
on a linear classification model trained on a given training set made by the
instructors.

Example usage from CLI:
 $ python3 project0.py --train_dir ~/path/to/train/dir/ --test_dir ~/path/to/test/dir/
--pred_file ~/path/to/pred/file

For help, run:
 $ python3 project0.py -h
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import argparse
import os

__author__ = "Josephine Yates; Philip Hartout"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch"

def main():
    # Load training and testing set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)
    df_test = pd.read_csv(FLAGS.test, header=0, index_col=0)

    # Process for modelling
    X_train, y_train, X_test = df_train.drop(['y'], axis=1).values,\
    df_train['y'].values, df_test.values

    # Modelling using linear regression
    model = LinearRegression().fit(X_train,y_train)

    # Make predictions and export to .csv
    predictions = model.predict(X_test)
    predictions_frame = pd.DataFrame(predictions, index=df_test.index, columns=["y"])
    predictions_frame.to_csv(FLAGS.pred, sep=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI args for folder and file directories')
    parser.add_argument("--train", "-tr", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--test", "-ts", type=str, required=True,
                        help="path to the CSV file containing the training data")
    parser.add_argument("--pred", "-pr", type=str, required=True,
                        help="path where the CSV file should be written")
    FLAGS = parser.parse_args()
    main()
