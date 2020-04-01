#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project2, subtasks 1 and 2, which aims to perform the following tasks:
 * Preprocess the ICU data
 * Resample these data when they are unbalanced
 * Scale the data
 * Fits SVMs
 * exports the results in a zip file

example usage from CLI:
 $ python3 subtask1and2.py --args

For help, run:
 $ subtask1and2.py -h

TODO:
    * Try using regular SVR to be able to use kernels
    * Clean up code to train one model per task ideally.
    * Write docstrings

Following Google style guide: http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Josephine Yates; Philip Hartout"
__email__ = "jyates@student.ethz.ch; phartout@student.ethz.ch"

import multiprocessing
import argparse
import logging

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TYPICAL_VALUES = {
    "pid": 15788.831218741774,
    "Time": 7.014398525927875,
    "Age": 62.07380889707818,
    "EtCO2": 32.88311356434632,
    "PTT": 40.09130983590656,
    "BUN": 23.192663516538175,
    "Lactate": 2.8597155076236422,
    "Temp": 36.852135856500034,
    "Hgb": 10.628207669881103,
    "HCO3": 23.488100167210746,
    "BaseExcess": -1.2392844571830848,
    "RRate": 18.154043187688046,
    "Fibrinogen": 262.496911351785,
    "Phosphate": 3.612519413287318,
    "WBC": 11.738648535345682,
    "Creatinine": 1.4957773156474896,
    "PaCO2": 41.11569643111729,
    "AST": 193.4448880402708,
    "FiO2": 0.7016656642357807,
    "Platelets": 204.66642639312448,
    "SaO2": 93.010527124635,
    "Glucose": 142.169406624713,
    "ABPm": 82.11727559995713,
    "Magnesium": 2.004148832962384,
    "Potassium": 4.152729193815373,
    "ABPd": 64.01471072970384,
    "Calcium": 7.161149186763874,
    "Alkalinephos": 97.79616327960757,
    "SpO2": 97.6634493216935,
    "Bilirubin_direct": 1.390723226703758,
    "Chloride": 106.26018538478121,
    "Hct": 31.28308971681893,
    "Heartrate": 84.52237068276303,
    "Bilirubin_total": 1.6409406684190786,
    "TroponinI": 7.269239936440605,
    "ABPs": 122.3698773806418,
    "pH": 7.367231494050988,
}


def load_data():
    """Loads data to three different dataframes.

    Returns:
        df_train, df_train_label, df_test (pandas.core.frame.DataFrame): three dataframes containing the
        training features, training labels and testing features respectively.

    """
    if FLAGS.nb_of_patients is not None:
        rows_to_load = (FLAGS.nb_of_patients * 12) + 1
    else:
        rows_to_load = None
    df_train = pd.read_csv(FLAGS.train_features, nrows=rows_to_load)
    df_train_label = pd.read_csv(FLAGS.train_labels, nrows=rows_to_load)
    df_test = pd.read_csv(FLAGS.test_features, nrows=rows_to_load)
    return df_train, df_train_label, df_test


# slower version - supports patient specific mean
def fill_na_with_average_patient_column(df, logger):
    """Fills NaNs with the average value of each column for each patient if available,
    otherwise column-wide entry

    Args:
        df (pandas.core.frame.DataFrame): data to be transformed
        logger (Logger): logger

    Returns:
        df (pandas.core.frame.DataFrame): dataframe containing the transformed data
    """
    columns = list(df.columns)
    for i, column in enumerate(columns):
        logger.info("{} column of {} columns processed".format(i + 1, len(columns)))
        # Fill na with patient average
        df[[column]] = df.groupby(["pid"])[column].transform(
            lambda x: x.fillna(x.mean())
        )

    # Fill na with overall column average for lack of a better option for now
    df = df.fillna(df.mean())
    if df.isnull().values.any():
        columns_with_na = df.columns[df.isna().any()].tolist()
        for column in columns_with_na:
            df[column] = TYPICAL_VALUES[column]
    return df


def fill_na_with_average_column(df):
    """Quick version of fill_na_with_average_patient_column - does not support patient average
    and results in loss of information.

    Note:
        Inserted dict with typical values as global var because running the script on parts of the data
    leads to errors associated with NaNs because there is not a single sample.

    Args:
        df (pandas.core.DataFrame): data to be transformed

    Returns:
        df (pandas.core.frame.DataFrame): dataframe containing the transformed data
    """

    df = df.fillna(df.mean(numeric_only=True))
    if df.isnull().values.any():
        columns_with_na = df.columns[df.isna().any()].tolist()
        for column in columns_with_na:
            df[column] = TYPICAL_VALUES[column]
    return df


def oversampling_strategies(X_train, y_train, strategy):
    """Function that uses a strategy to balance the dataset.

    Args:
        X_train (numpy.ndarray): array containing the features that need to be balanced
        y_train (numpy.ndarray): array containing the data labels
        strategy (int): sampling strategy to be adopted

    Returns:
        X_train_resampled, y_train_resampled (numpy.ndarray): resampled features and labels according
        to the chosen strategy
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

    identifiers = ["pid", "Time"]
    medical_tests = [
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
    vital_signs = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
    sepsis = ["LABEL_Sepsis"]

    logger.info("Preprocess training set")
    # Would be useful to distribute/multithread this part
    df_train_preprocessed = fill_na_with_average_patient_column(df_train, logger)

    # Cast training labels for these tasks
    df_train_label[medical_tests + vital_signs + sepsis] = df_train_label[
        medical_tests + vital_signs + sepsis
    ].astype(int)
    # Merging pids to make sure they map correctly.
    df_train_preprocessed_merged = pd.merge(
        df_train_preprocessed, df_train_label, how="left", left_on="pid", right_on="pid"
    )
    # Cast to arrays
    X_train = df_train_preprocessed_merged.drop(
        columns=identifiers + medical_tests + sepsis + vital_signs
    ).values
    # Create list with different label for each medical test
    logger.info("Creating a list of labels for each medical test")
    y_train_set_med = []
    for test in medical_tests:
        y_train_set_med.append(df_train_preprocessed_merged[test].values)
    y_train_sepsis = df_train_preprocessed_merged["LABEL_Sepsis"].values

    logger.info("Preprocess test set")
    df_test_preprocessed = fill_na_with_average_patient_column(df_train, logger)

    # Scale data to avoid convergence warning
    logger.info(f"Scaling data using {FLAGS.scaler}.")

    if FLAGS.scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    # Compute resampled data for all medical tests
    logger.info("Beginning sampling strategy for medical tests")
    X_train_resampled_set_med, y_train_resampled_set_med = get_sampling_medical_tests(
        logger, X_train, y_train_set_med, medical_tests, FLAGS.sampling_strategy
    )
    logger.info("Performing oversampling for sepsis.")
    # Can be called directly because there is only one label.
    X_train_resampled_sepsis, y_train_resampled_sepsis = oversampling_strategies(
        X_train, y_train_sepsis, FLAGS.sampling_strategy
    )

    logger.info("Beginning modelling process.")

    # Hyperparameter grid specification
    param_grid_linear = {
        "penalty": ["l1", "l2"],
        "loss": ["squared_hinge"],
        "dual": [False],
        "tol": [0.0001],
        "C": np.linspace(0.1, 10, num=3),
        "multi_class": ["ovr"],
        "fit_intercept": [False],
        "intercept_scaling": [
            1
        ],  # From docs: To lessen the effect of regularization on synthetic feature weight
        # (and therefore on the intercept) intercept_scaling has to be increased.
        "class_weight": [
            None
        ],  # Sampling strategy already takes care of this, otherwise add option "balanced"
        # to see the effect
        "verbose": [0],  # Doesn't work well given the gridsearch as per docs
        "random_state": [42],  # Because we <3 Douglas Adams.
        "max_iter": [-1],  # Stopping criterion is given by the tol hyperparameter.
    }

    param_grid_non_linear = {
        "C": np.linspace(0.1, 10, num=3),
        "kernel": ["rbf", "sigmoid"],
        "degree": range(
            1, 4
        ),  # This really dictates the runtime of the algorithm, to tune carefully.
        "gamma": np.linspace(0.1, 10, num=3),  # for poly or rbf kernel
        "coef0": [0],
        "shrinking": [True],
        "probability": [False],
        "tol": [0.001],
        "cache_size": [200],
        "class_weight": [None],
        "verbose": [False],
        "max_iter": [1000],
        "decision_function_shape": [
            "ovo"
        ],  # That's because we train one classifer per test.
        "random_state": [42],
    }

    # CV GridSearch with different regularization parameters
    logger.info("Perform gridsearch for linear SVM on medical tests.")
    gridsearch_l_svm_medical_tests_models, scores_l_svm_medical_tests_models = get_models_medical_tests(
        X_train_resampled_set_med,
        y_train_resampled_set_med,
        logger,
        medical_tests,
        param_grid_linear,
        "gridsearch_linear",
    )

    logger.info("Perform gridsearch for non-linear SVM on medical tests.")
    gridsearch_nl_svm_medical_tests_models, scores_nl_svm_medical_tests_models = get_models_medical_tests(
        X_train_resampled_set_med,
        y_train_resampled_set_med,
        logger,
        medical_tests,
        param_grid_non_linear,
        "gridsearch_non_linear",
    )

    best_model_medical_tests = determine_best_model_medical_test(
        gridsearch_l_svm_medical_tests_models,
        gridsearch_nl_svm_medical_tests_models,
        scores_l_svm_medical_tests_models,
        scores_nl_svm_medical_tests_models,
        medical_tests,
        logger,
    )

    logger.info("Perform gridsearch for linear SVM on sepsis.")
    gridsearch_l_sepsis_model, scores_l_sepsis_model = get_model_sepsis(
        X_train_resampled_sepsis,
        y_train_resampled_sepsis,
        logger,
        param_grid_linear,
        "gridsearch_linear",
    )

    logger.info("Perform gridsearch for non-linear SVM on sepsis.")
    gridsearch_nl_svm_sepsis_models, scores_nl_svm_sepsis_models = get_model_sepsis(
        X_train_resampled_sepsis,
        y_train_resampled_sepsis,
        logger,
        param_grid_non_linear,
        "gridsearch_non_linear",
    )

    best_model_sepsis = determine_best_model_sepsis(
        [scores_l_sepsis_model, scores_nl_svm_sepsis_models],
        [gridsearch_l_sepsis_model, gridsearch_nl_svm_sepsis_models],
        logger,
    )

    X_test = df_test_preprocessed.drop(columns=identifiers).values

    # get the unique test ids of patients
    test_pids = np.unique(df_test_preprocessed[["pid"]].values)
    logger.info("Fetch predictions.")
    medical_test_predictions = get_medical_test_predictions(
        X_test, test_pids, best_model_medical_tests, medical_tests
    )
    sepsis_predictions = get_sepsis_predictions(
        X_test, test_pids, best_model_sepsis, sepsis
    )
    medical_test_predictions.index.names = ["pid"]
    sepsis_predictions.index.names = ["pid"]
    predictions = pd.merge(
        medical_test_predictions,
        sepsis_predictions,
        how="left",
        left_on="pid",
        right_on="pid",
    )

    logger.info("Export predictions DataFrame to a zip file")
    # Export pandas dataframe to zip archive.
    predictions.to_csv(
        FLAGS.predictions, index=False, float_format="%.3f", compression="zip"
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
        help="path to the CSV file containing the training \
                        features",
    )

    parser.add_argument(
        "--test_features",
        "-test",
        type=str,
        required=True,
        help="path to the CSV file containing the testing \
                            features",
    )

    parser.add_argument(
        "--train_labels",
        "-train_l",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
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
    logging.basicConfig(level=logging.DEBUG,
                        filename="script_status.log")

    logger = logging.getLogger("IML-P2-T1T2")

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] "
                                                  "%(message)s"))
    logger.addHandler(stream_handler)

    main(logger)
