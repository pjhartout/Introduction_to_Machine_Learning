#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project2, which aims to perform the following tasks:
 * Resample these data when they are unbalanced
 * Scale the data
 * Fits SVM/SVR
 * exports the results in a zip file

example usage from CLI:
 $ python3 subtasks_ANN.py --train_features /path/to/preprocessed_train_features.csv
     --train_labels /path/to/preprocessed_train_labels.csv
     --test_features /path/to/preprocessed_test_features.csv
     --predictions /path/to/preprocessed_predictions_subtask3.zip
     --scaler minmax

For help, run:
 $ subtasks_SVM.py -h



Following Google style guide: http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Josephine Yates; Philip Hartout"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch"

import multiprocessing
import argparse
import logging
import os
import shutil
import sys
import zipfile
import time
import sys

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.svm import SVR
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

IDENTIFIERS = ["pid"]
MEDICAL_TESTS = [
    "LABEL_BaseExcess",
    "LABEL_Fibrinogen",
    "LABEL_AST",
    "LABEL_Alkalinephos",
    "LABEL_Bilirubin_total",
    "LABEL_Lactate",
    "LABEL_TroponinI",
    "LABEL_SaO2",
    "LABEL_Bilirubin_direct",
    "LABEL_EtCO2",
]
VITAL_SIGNS = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
SEPSIS = ["LABEL_Sepsis"]


def sigmoid_f(x):
    """To get predictions as confidence level, the model predicts for all 12 sets of measures for
    each patient a distance to the hyperplane ; it is then transformed into a confidence level using
    the sigmoid function ; the confidence level reported is the mean of all confidence levels for a
    single patient

    Args:
        x (float): input of the sigmoid function

    Returns:
       float: result of the sigmoid computation.

    """
    return 1 / (1 + np.exp(-x))


def load_data():
    """Loads the preprocessed data to three different dataframes.

    Returns:
        df_train, df_train_label, df_test (pandas.core.frame.DataFrame): three dataframes
        containing the
        preprocessed training features, training labels and testing features respectively.

    """
    df_train = pd.read_csv(FLAGS.train_features, nrows=FLAGS.nb_of_patients)
    df_train_label = pd.read_csv(FLAGS.train_labels, nrows=FLAGS.nb_of_patients)
    df_test = pd.read_csv(FLAGS.test_features)
    return df_train, df_train_label, df_test


def data_formatting(df_train, df_train_label, logger):
    """Function takes data in for formatting

    Args:
        df_train (pandas.core.DataFrame): preprocessed training features
        df_train_label (pandas.core.DataFrame): preprocessed training labels
        logger (Logger): logger

    Returns:
        X_train (np.ndarray): (n_samples, n_features) array containing features
        y_train_vital_signs (np.ndarray): (n_samples, n_features) array labels

        transform outputs later on when scaling back predictions for interpretability
    """

    # Cast to arrays
    X_train = df_train.drop(columns=IDENTIFIERS).values

    # Create list with different label for each medical test
    logger.info("Creating a list of labels for each medical test")
    y_train_medical_tests = []
    for test in MEDICAL_TESTS:
        y_train_medical_tests.append(df_train_label[test].astype(int).values)

    # Create list with different label for sepsis
    logger.info("Creating a list of labels for each medical test")
    y_train_sepsis = []
    for sepsis in SEPSIS:
        y_train_sepsis.append(df_train_label[sepsis].astype(int).values)

    # Create list with different label for each vital sign
    logger.info("Creating a list of labels for each vital sign")
    y_train_vital_signs = []
    for sign in VITAL_SIGNS:
        y_train_vital_signs.append(df_train_label[sign].astype(int).values)

    # Scale data to avoid convergence warning
    logger.info(f"Scaling data using {FLAGS.scaler}.")

    if FLAGS.scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    return X_train, y_train_medical_tests, y_train_sepsis, y_train_vital_signs


def oversampling_strategies(X_train, y_train, strategy):
    """Function that uses a strategy to balance the dataset.

    Args:
        X_train (numpy.ndarray): array containing the features that need to be balanced
        y_train (numpy.ndarray): array containing the data labels
        strategy (int): sampling strategy to be adopted

    Returns:
        X_train_resampled, y_train_resampled (numpy.ndarray): resampled features and labels
        according to the chosen strategy
    """

    # Oversampling methods
    if strategy == "adasyn":
        sampling_method = ADASYN()
    if strategy == "smote":
        sampling_method = SMOTE()

    # Undersampling methods
    if strategy == "clustercentroids":
        sampling_method = ClusterCentroids(random_state=42)
    if strategy == "random":
        sampling_method = RandomUnderSampler(random_state=0, replacement=True)

    X_train_resampled, y_train_resampled = sampling_method.fit_sample(X_train, y_train)

    return X_train_resampled, y_train_resampled


def get_sampling_medical_tests(logger, X_train, y_train_set_med, sampling_strategy):
    """Resamples the data required for each medical tests.

    Args:
        logger (Logger): logger
        X_train (np.ndarray): array that contains the features that need to be resampled
        y_train_set_med (np.ndarray): array that contains the labels that need to be balanced.
        sampling_strategy (str): Sampling strategy to be adopted

    Returns:
        X_train_resampled_set_med (list): list of np.ndarray containing the resampled dataset for
        each of the test
        y_train_resampled_set_med (list): list of np.ndarray containing the resampled labels for
        each of the test
    """
    X_train_resampled_set_med, y_train_resampled_set_med = (
        [0] * len(y_train_set_med),
        [0] * len(y_train_set_med),
    )
    number_of_tests = len(y_train_set_med)
    for i in range(number_of_tests):
        X_train_resampled_set_med[i], y_train_resampled_set_med[
            i
        ] = oversampling_strategies(X_train, y_train_set_med[i], sampling_strategy)
        logger.info(
            "Performing oversampling for {} of {} medical tests ({}).".format(
                i, number_of_tests, MEDICAL_TESTS[i]
            )
        )
    return X_train_resampled_set_med, y_train_resampled_set_med


def get_sampling_sepsis(logger, X_train, y_train_sepsis, sampling_strategy):
    """Resamples the data required for sepsis

    Args:
        logger (Logger): logger
        X_train (np.ndarray): array that contains the features that need to be resampled
        y_train_sepsis (np.ndarray): array that contains the labels that need to be balanced.
        sampling_strategy (str): Sampling strategy to be adopted

    Returns:
        X_train_resampled_sepsis (np.ndarray): contains the resampled dataset for sepsis
        y_train_resampled_set_med (np.ndarray): contains the resampled labels for sepsis
    """

    logger.info("Performing oversampling for sepsis")
    X_train_resampled_sepsis, y_train_resampled_sepsis = oversampling_strategies(
        X_train, y_train_sepsis[0], sampling_strategy
    )
    return ([X_train_resampled_sepsis], [y_train_resampled_sepsis])


@ignore_warnings(category=ConvergenceWarning)
def get_models(
    X_train_resampled_set, y_train_resampled_set, logger, param_grid, subtask
):
    """

    Args:
        X_train_resampled_set (list): list containing the arrays containing the training data for
        each test.
        y_train_resampled_set (list): list containing the arrays containing the training labels for
        each test.
        logger (Logger): logger
        param_grid (list): parameter grid to be used for the gridsearch
        subtask (int): subtask performed

    Returns:
        svm_models (list): list of fitted models (best gridsearch result for each medical test).
        scores (list): corresponding list of cross-validation scores as given by the gridsearch.
    """
    svm_models = []
    scores = []
    topred = (
        MEDICAL_TESTS if subtask == 1 else (SEPSIS if subtask == 2 else VITAL_SIGNS)
    )
    if subtask == 1 or subtask == 3:
        for i, test in enumerate(topred):
            logger.info(f"Starting iteration for test {test}")
            cores = multiprocessing.cpu_count() - 2
            if subtask == 1:
                gs_svm = GridSearchCV(
                    estimator=SVC(),
                    param_grid=param_grid,
                    n_jobs=cores,
                    scoring="roc_auc",
                    cv=FLAGS.k_fold,
                    verbose=0,
                )
            elif subtask == 3:
                gs_svm = GridSearchCV(
                    estimator=SVR(),
                    param_grid=param_grid,
                    n_jobs=cores,
                    scoring="r2",
                    cv=FLAGS.k_fold,
                    verbose=0,
                )
            gs_svm.fit(X_train_resampled_set[i], y_train_resampled_set[i])
            svm_models.append(gs_svm.best_estimator_)
            scores.append(gs_svm.best_score_)
    elif subtask == 2:
        cores = multiprocessing.cpu_count() - 2
        gs_svm = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            n_jobs=cores,
            scoring="roc_auc",
            cv=FLAGS.k_fold,
            verbose=0,
        )
        gs_svm.fit(X_train_resampled_set[0], y_train_resampled_set[0])
        svm_models.append(gs_svm.best_estimator_)
        scores.append(gs_svm.best_score_)

    return svm_models, scores


def get_all_predictions(X_test, test_pids, models):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1
    (resp 0), the more confidently) the sample belongs to class 1 (resp 0).

    Args:
        X_test (np.ndarray): array of preprocessed test values
        test_pids (np.ndarray): array of patient ids in test set
        models (list): list of models for each of the tests for all subtasks (in the order of subtasks)

    Returns:
        df_pred (pandas.core.DataFrame): contains the predictions made by each of the models for
        their respective tests, containing for each patient id the predicted label as a confidence
        level.
    """
    df_pred = pd.DataFrame()

    for i, test in enumerate(MEDICAL_TESTS):
        # decision_function returns the distance to the hyperplane
        y_conf = models[i].decision_function(X_test)
        # compute the predictions as confidence levels, ie using sigmoid function instead of sign
        # function
        y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
        df = pd.DataFrame({test: y_pred}, index=test_pids)
        df_pred = pd.concat([df_pred, df], axis=1)
    y_conf = models[len(MEDICAL_TESTS)].decision_function(X_test)
    y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
    df = pd.DataFrame({SEPSIS[0]: y_pred}, index=test_pids)
    df_pred = pd.concat([df_pred, df], axis=1)
    for i, sign in enumerate(VITAL_SIGNS):
        # decision_function returns the distance to the hyperplane
        y_conf = models[len(MEDICAL_TESTS) + i].predict(X_test)
        df = pd.DataFrame({sign: y_pred}, index=test_pids)
        df_pred = pd.concat([df_pred, df], axis=1)
    return df_pred


def main(logger):
    """Primary function reading, preprocessing and modelling the data

    Args:
        logger (Logger): logger to get information about the status of the script when running

    Returns:
        None
    """

    logger.info("Loading data")
    df_train, df_train_label, df_test = load_data()
    logger.info("Finished Loading data")

    X_train, y_train_medical_tests, y_train_sepsis, y_train_vital_signs = data_formatting(
        df_train, df_train_label, logger
    )

    # Compute resampled data for all medical tests
    logger.info("Beginning sampling strategy for medical tests")
    X_train_resampled_set_med, y_train_resampled_set_med = get_sampling_medical_tests(
        logger, X_train, y_train_medical_tests, FLAGS.sampling_strategy
    )
    logger.info("Performing oversampling for SEPSIS.")
    X_train_resampled_sepsis, y_train_resampled_sepsis = get_sampling_sepsis(
        logger, X_train, y_train_sepsis, FLAGS.sampling_strategy
    )

    logger.info("Beginning modelling process.")

    # Hyperparameter grid specification

    param_grid_non_linear = {
        "C": np.linspace(10, 100, num=3),
        "kernel": ["rbf", "sigmoid"],
        "gamma": np.linspace(5, 10, num=3),  # for poly or rbf kernel
        "coef0": [0],
        "shrinking": [True],
        "probability": [False],
        "cache_size": [1000],
        "class_weight": [{0: 0.2, 1: 0.8}],
        "verbose": [2],
        "decision_function_shape": ["ovo"],  # only binary variables are set
        "random_state": [42],
        "max_iter": [2000],
    }

    param_grid_SVR = {
        "kernel": ["poly", "rbf", "sigmoid"],
        "degree": np.arange(1, 4, 1),
        "gamma": np.linspace(0.1, 10, num=3),
        "coef0": [0],
        "tol": [0.001],
        "C": np.linspace(0.1, 10, num=3),
        "epsilon": [0.1],
        "shrinking": [True],
        "cache_size": [1000],
        "verbose": [2],
        "max_iter": [1000],
    }

    # CV GridSearch with different regularization parameters

    logger.info("Perform gridsearch for non-linear SVM on medical tests.")
    gridsearch_nl_svm_medical_tests_models, scores_nl_svm_medical_tests_models = get_models(
        X_train_resampled_set_med,
        y_train_resampled_set_med,
        logger,
        param_grid_non_linear,
        subtask=1,
    )
    X_test = df_test.drop(columns=IDENTIFIERS).values
    logger.info(
        f"Nonlinear SVM results medical tests {scores_nl_svm_medical_tests_models}"
    )
    best_model_medical_tests = gridsearch_nl_svm_medical_tests_models

    logger.info("Perform gridsearch for non-linear SVM on SEPSIS.")
    gridsearch_nl_svm_sepsis_models, scores_nl_svm_sepsis_models = get_models(
        X_train_resampled_sepsis,
        y_train_resampled_sepsis,
        logger,
        param_grid_non_linear,
        subtask=2,
    )
    logger.info(f"Nonlinear SVM results sepsis {scores_nl_svm_sepsis_models}")

    best_model_sepsis = gridsearch_nl_svm_sepsis_models

    logger.info("Perform gridsearch for non-linear SVR on VITAL SIGNS.")
    gridsearch_nl_svr_vital_signs_models, scores_nl_svr_vital_signs_models = get_models(
        [X_train] * len(VITAL_SIGNS),
        y_train_vital_signs,
        logger,
        param_grid_SVR,
        subtask=3,
    )
    logger.info(f"Nonlinear SVM results vital signs {scores_nl_svr_vital_signs_models}")
    best_model_vital_signs = gridsearch_nl_svr_vital_signs_models

    X_test = df_test.drop(columns=IDENTIFIERS).values
    all_models = best_model_medical_tests + best_model_sepsis + best_model_vital_signs
    # get the unique test ids of patients
    test_pids = np.unique(df_test[["pid"]].values)
    logger.info("Fetch predictions.")
    predictions = get_all_predictions(X_test, test_pids, all_models)
    predictions.index.names = ["pid"]

    logger.info("Export predictions DataFrame to a zip file")
    predictions.to_csv(
        FLAGS.predictions,
        index=False,
        float_format="%.3f",
        compression=dict(method="zip", archive_name="predictions.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI args for folder and file \
    directories"
    )

    parser.add_argument(
        "--train_features",
        "-train_f",
        type=str,
        required=True,
        help="path to the CSV file containing the preprocessed training \
                        features",
    )

    parser.add_argument(
        "--test_features",
        "-test",
        type=str,
        required=True,
        help="path to the CSV file containing the preprocessed testing \
                            features",
    )

    parser.add_argument(
        "--train_labels",
        "-train_l",
        type=str,
        required=True,
        help="path to the CSV file containing the preprocessed training \
                                labels",
    )

    parser.add_argument(
        "--predictions",
        "-pred",
        type=str,
        required=True,
        help="path to the zip file containing the \
                                predictions",
    )

    parser.add_argument(
        "--nb_of_patients",
        "-nb_pat",
        type=int,
        required=False,
        help="Number of patients to consider in run. If not specified, then consider all patients",
    )

    parser.add_argument(
        "--sampling_strategy",
        "-samp",
        type=str,
        required=True,
        help="Sampling strategy to adopt to overcome the imbalanced dataset problem"
        "any of adasyn, smote, clustercentroids or random.",
        choices=["adasyn", "smote", "clustercentroids", "random"],
    )

    parser.add_argument(
        "--scaler",
        "-scale",
        type=str,
        required=True,
        help="Scaler to be used to transform the data.",
        choices=["minmax", "standard"],
    )

    parser.add_argument(
        "--k_fold",
        "-k",
        type=int,
        required=True,
        help="k to perform k-fold cv in the gridsearch",
    )

    FLAGS = parser.parse_args()

    # clear logger.
    logging.basicConfig(level=logging.DEBUG, filename="script_status_SVM.log")

    logger = logging.getLogger("IML-P2-SVM")

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] "
            "%(message)s"
        )
    )
    logger.addHandler(stream_handler)

    main(logger)
