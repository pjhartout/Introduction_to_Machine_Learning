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
from sklearn.svm import SVR
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

IDENTIFIERS = ["pid", "Time"]
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


def load_data():
    """Loads data to three different dataframes.

    Returns:
        df_train, df_train_label, df_test (pandas.core.frame.DataFrame): three dataframes
        containing the
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
        logger.info(
            "{} column of {} columns processed".format(i + 1, len(columns))
        )
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


def data_preprocessing(df_train, df_train_label, df_test, logger):
    logger.info("Preprocess training set")
    # Would be useful to distribute/multithread this part
    df_train_preprocessed = fill_na_with_average_patient_column(
        df_train, logger
    )

    logger.info("Perform projection to select only vital signs labels")
    df_train_label = df_train_label[["pid"] + VITAL_SIGNS]

    # Merging pids to make sure they map correctly.
    df_train_preprocessed_merged = pd.merge(
        df_train_preprocessed,
        df_train_label,
        how="left",
        left_on="pid",
        right_on="pid",
    )
    # Cast to arrays
    X_train = df_train_preprocessed_merged.drop(
        columns=IDENTIFIERS + VITAL_SIGNS
    ).values
    # Create list with different label for each medical test
    logger.info("Creating a list of labels for each medical test")
    y_train_vital_signs = []
    for sign in VITAL_SIGNS:
        y_train_vital_signs.append(df_train_preprocessed_merged[sign].values)

    logger.info("Preprocess test set")
    df_test_preprocessed = fill_na_with_average_patient_column(df_test, logger)
    # Scale data to avoid convergence warning
    logger.info(f"Scaling data using {FLAGS.scaler}.")

    if FLAGS.scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    return X_train, y_train_vital_signs, df_test_preprocessed


def get_vital_signs_models(X_train, y_train_vital_signs, param_grid):
    svr_models = []
    scores = []
    for i, test in enumerate(VITAL_SIGNS):
        logger.info(f"Starting iteration for test {test}")
        svr = SVR()
        cores = multiprocessing.cpu_count() - 2
        tuned_svr = GridSearchCV(
            estimator=svr,
            param_grid=param_grid,
            n_jobs=cores,
            scoring="r2",
            cv=FLAGS.k_fold,
            verbose=0,
        )
        tuned_svr.fit(X_train, y_train_vital_signs[i])
        svr_models.append(tuned_svr.best_estimator_)
        scores.append(tuned_svr.best_score_)

    return svr_models, scores


def get_vital_signs_predictions(X_test, test_pids, models):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1
    (resp 0), the more confidently) the sample belongs to class 1 (resp 0).

    Args:
        X_test (np.ndarray): array of preprocessed test values
        test_pids (np.ndarray): array of patient ids in test set
        svm_models (list): list of models for each of the medical tests

    Returns:
        df_pred (pandas.core.DataFrame): contains the predictions made by each of the models for
        their respective tests, containing for each patient id the predicted label as a confidence
        level.
    """
    df_pred = pd.DataFrame()

    for i, test in enumerate(VITAL_SIGNS):
        # Compute prediction
        y_pred = models[i - 1].predict(X_test)

        y_mean = [np.mean(y_pred[i : i + 12]) for i in range(len(test_pids))]
        df = pd.DataFrame({test: y_mean}, index=test_pids)
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

    X_train, y_train_vital_signs, df_test_preprocessed = data_preprocessing(
        df_train, df_train_label, df_test, logger
    )

    logger.info("Beginning modelling process.")

    # train model here
    param_grid = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": np.arange(1, 4, 1),
        "gamma": np.linspace(0.1, 10, num=3),
        "coef0": [0],
        "tol": [0.001],
        "C": np.linspace(0.1, 10, num=3),
        "epsilon": [0.1],
        "shrinking": [True],
        "cache_size": [1000],
        "verbose": [False],
        "max_iter": [1000],
    }
    vital_signs_models, vital_signs_models_scores = get_vital_signs_models(
        X_train, y_train_vital_signs, param_grid
    )

    # get the unique test ids of patients
    test_pids = np.unique(df_test_preprocessed[IDENTIFIERS].values)
    logger.info("Fetch predictions.")
    X_test = df_test_preprocessed.drop(columns=IDENTIFIERS).values
    vital_signs_predictions = get_vital_signs_predictions(
        X_test, test_pids, vital_signs_models
    )

    logger.info("Export predictions DataFrame to a zip file")
    # Export pandas dataframe to zip archive.
    vital_signs_predictions.to_csv(
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
    logging.basicConfig(
        level=logging.DEBUG, filename="script_status_subtask3.log"
    )

    logger = logging.getLogger("IML-P2-T1T2")

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        )
    )
    logger.addHandler(stream_handler)

    main(logger)
