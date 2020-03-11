#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project1, task b, which endeavours to perform ridge
regression using feature transformation

example usage from CLI:
 $ python3 task_1b.py --train ~/path/to/train/dir/
                      --weights ~/path/to/weights/file

For help, run:
 $ python3 task_1b.py -h

Following Google style guide: http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Josephine Yates; Philip Hartout; Flavio Rump"
__email__ = (
    "jyates@student.ethz.ch; phartout@student.ethz.ch; flrump@student.ethz.ch"
)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def plot_error_model(alphas, mean_errors):
    """This function plots the mean error of CV vs the regularization parameter
    of the model being evaludated.

    Args:
        alphas (numpy.ndarray): regularization parameters tried out
        mean_errors (numpy.ndarray): result of average error over cross
            validation for different alphas

    Returns:
        None
    """
    plt.plot(alphas, mean_errors)
    plt.title("Mean cv error vs regularization parameter")
    plt.xlabel("Alpha, regularization parameter")
    plt.ylabel("Mean error")
    plt.show()


def linear(input_vector):
    """Returns the same array as passed

    Args:
        x (numpy.ndarray): vector or array of numbers to be passed

    Returns:
        The same array
    """
    return input_vector


def constant(input_vector):
    """Returns a vector containing ones the dimensions of the input array

    Args:
        x (numpy.ndarray): vector or array containing data

    Returns:
        An array of ones of the same length as the input
    """
    return np.ones((input_vector.shape[0],)).reshape(-1, 1)


def main():
    """Primary function reading, preprocessing, transforming, scaling and
    transforming the data. Then fits a series of Ridge regression on it.

    Args:
        None

    Returns:
        None
    """
    # Load training set
    df_train = pd.read_csv(FLAGS.train, header=0, index_col=0)

    # Process for modelling
    x_train, y_train = df_train.drop(["y"], axis=1), df_train["y"].values

    # Declare scaler
    scaler = StandardScaler(with_mean=True, with_std=True)

    # Declare transformations to be done
    transformations = [linear, np.square, np.exp, np.cos, constant]
    weights_list = []

    # loop through data transformations
    for transformation in transformations:
        # Transform the data
        transformer = FunctionTransformer(transformation)
        x_train_trans = transformer.transform(x_train)

        # Scale the data
        x_train_trans_scaled = scaler.fit_transform(x_train_trans)

        # Specify the alpha range to use
        alpha_range = np.arange(0.1, 10, 0.1)

        # Fit the regression with RidgeCV
        reg = RidgeCV(
            fit_intercept=False, alphas=alpha_range, store_cv_values=True
        ).fit(x_train_trans_scaled, y_train)

        # Append coefs to the list of weights
        weights_list.append(reg.coef_)

        # Compute errors for plotting and analysing alpha values
        errors = reg.cv_values_
        mean_errors = [np.mean(errors[:, i]) for i in range(errors.shape[1])]
        plot_error_model(alpha_range, mean_errors)
        print(f"Selected value for alpha is: {reg.alpha_}")

    # flatten list coefficients
    weights_list = [item for sublist in weights_list for item in sublist]

    # export to .csv
    weight_df = pd.DataFrame(weights_list)
    weight_df.to_csv(
        FLAGS.weights, sep=" ", index=False, header=False, float_format="%.5f"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CLI args for folder and file \
    directories"
    )

    parser.add_argument(
        "--train",
        "-tr",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
                        data",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="path where the CSV file where weights should be \
                        written",
    )

    FLAGS = parser.parse_args()
    main()
