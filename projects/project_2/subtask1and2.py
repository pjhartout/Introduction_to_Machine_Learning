#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for project2, subtasks 1 and 2, which aims to perform the following tasks:
 * 

example usage from CLI:
 $ python3 subtask1and2.py --args

For help, run:
 $ subtask1and2.py -h

 TODO: Try using regular SVR to be able to use kernels

Following Google style guide: http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Josephine Yates; Philip Hartout"
__email__ = (
    "jyates@student.ethz.ch; phartout@student.ethz.ch"
)

import multiprocessing
import argparse
import logging

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from random import sample


# slower version - supports patient specific mean
def fill_na_with_average_patient_column(df_train, logger):
    columns = list(df_train.columns)
    del columns[0:2]

    df_train_preprocessed = df_train

    for i,column in enumerate(columns):
        logger.info("{} column of {} columns processed".format(i+1,len(columns)))
        # Fill na with patient average 
        df_train_preprocessed[[column]] = df_train_preprocessed.groupby(['pid'])[column]        .transform(lambda x: x.fillna(x.mean()))
        
    # Fill na with overall column average for lack of a better option for now
    df.fillna(df.mean())
    return df_train_preprocessed


# quick version - does not support patient average
def fill_na_with_average_column(df):
    return df.fillna(df.mean())


def oversampling_strategies(X_train, y_train, strategy="adasyn"):
    # Oversampling methods
    if strategy=="adasyn":
        sampling_method = ADASYN()
    if strategy=="smote":
        sampling_method = SMOTE()
    
    # Undersampling methods
    if strategy=="clustercentroids":
        sampling_method = ClusterCentroids(random_state=42)
    if strategy=="random":
        sampling_method = RandomUnderSampler(random_state=0, replacement=True)
        
    X_train_resampled, y_train_resampled = sampling_method.fit_sample(X_train, y_train)
    
    print(sorted(Counter(y_train_resampled).items()))
    
    return X_train_resampled, y_train_resampled


def get_random_sample(X_train_resampled_set,y_train_resampled_set,size=100):
    """Sample at random datapoints from the resampled datasets for each medical test
    
    Parameters: 
        X_train_resampled_set = np.array, set of size # of medical tests, with X_train for each
        y_train_resampled_set = np.array, set of size # of medical tests, with y_train for each
                                size = int, size of selected sample
    Returns:
        X_train_rd_set,y_train_rd_set : np.array, reduced sample sets where xxx_train_rd_set[i] is the reduced
                                            set for medical test i
    """
    X_train_rd_set,y_train_rd_set = [],[]
    for i,test in enumerate(medical_tests):
        ind = sample(range(len(X_train_resampled_set[i])),size)
        X_train_rd_set.append(X_train_resampled_set[i][ind])
        y_train_rd_set.append(y_train_resampled_set[i][ind])
    return np.array(X_train_rd_set),np.array(y_train_rd_set)


def get_models_medical_tests(X_train_resampled_set,y_train_resampled_set, medical_tests, alpha=10, param_grid={"C": [1,10]}, typ="naïve", reduced=True, size=100):
    """Function to obtain models for every set of medical test, either naïve or using CV Gridsearch
    
        Parameters: X_train_resampled_set = np.array, set of size # of medical tests, with X_train for each
                    y_train_resampled_set = np.array, set of size # of medical tests, with y_train for each
                    alpha = float (for naïve) regularization parameter, ignored if typ is not naive
                    param_grid = dict (for gridsearch), dictionary of parameters to search over, ignored if typ is not gridsearch
                    typ = str in ["naïve","gridsearch","naive_non_lin","gridsearch_non_lin"], default "naïve", how the task is performed
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    size = int, size of selected sample, ignored if reduced == False
        Returns:
                svr_models = list of Linear SVR models for each medical test, where svr_models[i] is the fitted 
                            model (best estimator in the case of gridsearch) for medical_test[i]
    """
    assert typ in ["naive","gridsearch","naive_non_lin","gridsearch_non_lin"], "typ must be in ['naive','gridsearch','naive_non_lin','gridsearch_non_lin']"
    # if reduced:
    #     X_train_resampled_set, y_train_resampled_set = get_random_sample(X_train_resampled_set,y_train_resampled_set,size=size)
    svm_models = []
    for i,test in enumerate(medical_tests):
        print("Starting iteration for test {}".format(test))
        if typ=="naive":
            # setting dual to false because n_samples>n_features
            lin_svm = LinearSVC(C=alpha,dual=False)
            lin_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            svm_models.append(lin_svm)
        elif typ=="gridsearch":
            cores=multiprocessing.cpu_count()-2
            gs_svm = GridSearchCV(estimator=LinearSVC(dual=False),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=0)
            gs_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
            svm_models.append(gs_svm.best_estimator_)
        elif typ=="naive_non_lin":
            lin_svm = SVC(C=alpha)
            lin_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            svm_models.append(lin_svm)
        elif typ=="gridsearch_non_lin":
            cores=multiprocessing.cpu_count()-2
            gs_svm = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=0)
            gs_svm.fit(X_train_resampled_set[i],y_train_resampled_set[i])
            print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
            svm_models.append(gs_svm.best_estimator_)
    return svm_models


def get_model_sepsis(X_train_resampled,y_train_resampled, alpha=10, param_grid={"C": [1,10]}, typ="naïve", reduced=True, size=100):
    svm = LinearSVC()
    assert typ in ["naive","gridsearch","naive_non_lin","gridsearch_non_lin"], "typ must be in ['naive','gridsearch','naive_non_lin','gridsearch_non_lin']"
    if reduced:
        ind = sample(range(len(X_train_resampled)),size)
        X_train_resampled, y_train_resampled = X_train_resampled[ind],y_train_resampled[ind]
    if typ=="naive":
        # setting dual to false because n_samples>n_features
        svm = LinearSVC(C=alpha,dual=False)
        svm.fit(X_train_resampled,y_train_resampled)
    elif typ=="gridsearch":
        cores=multiprocessing.cpu_count()-2
        gs_svm = GridSearchCV(estimator=LinearSVC(dual=False),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=0)
        gs_svm.fit(X_train_resampled,y_train_resampled)
        print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
        svm = gs_svm.best_estimator_
    elif typ=="naive_non_lin":
        svm = SVC(C=alpha)
        svm.fit(X_train_resampled,y_train_resampled)
    elif typ=="gridsearch_non_lin":
        cores=multiprocessing.cpu_count()-2
        gs_svm = GridSearchCV(estimator=SVC(),param_grid=param_grid,n_jobs=cores,scoring="roc_auc",verbose=0)
        gs_svm.fit(X_train_resampled,y_train_resampled)
        print("The estimated auc roc score for this estimator is {}, with alpha = {}".format(gs_svm.best_score_,gs_svm.best_params_))
        svm = gs_svm.best_estimator_
    return svm


def sigmoid_f(x):
    """To get predictions as confidence level, the model predicts for all 12 sets of measures for each patient a
    distance to the hyperplane ; it is then transformed into a confidence level using the sigmoid function ; the
    confidence level reported is the mean of all confidence levels for a single patient
    """
    return 1/(1 + np.exp(-x))


def get_predictions(X_test,test_pids,svm_models,medical_tests,reduced=False,nb_patients=100):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1 (resp 0), the more confidently)
            the sample belongs to class 1 (resp 0).
        Parameters: X_test = np.array, set of preprocessed test values
                    test_pids = np.array, unique set of patient ids in test set
                    svm_models = list, fitted svm models to training set 
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    nb_patients = int, size of number of patients selected, ignored if reduced == False
        Returns:
                df_pred = pd.DataFrame, dataframe containing for each patient id the predicted label as a confidence level
    """
    df_pred = pd.DataFrame()
    if reduced:
        # sample at random nb_patients patients 
        test_pids = sample(list(test_pids),nb_patients)
        X_test = X_test[test_pids]
    for i,test in enumerate(medical_tests):
        # decision_function returns the distance to the hyperplane
        print(svm_models[i])
        y_conf = svm_models[i].decision_function(X_test)
        # compute the predictions as confidence levels, ie using sigmoid function instead of sign function
        y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
        # use the mean of the computation for each patient as overall confidence level 
        y_mean = [np.mean(y_pred[i:i+12]) for i in range(len(test_pids))]
        df = pd.DataFrame({test: y_mean},index=test_pids)
        df_pred = pd.concat([df_pred,df], axis=1)
    return df_pred


def get_sepsis_predictions(X_test,test_pids,svm,sepsis,reduced=False,nb_patients=100):
    """Function to obtain predictions for every model, as a confidence level : the closer to 1 (resp 0), the more confidently)
            the sample belongs to class 1 (resp 0).
        Parameters: X_test = np.array, set of preprocessed test values
                    test_pids = np.array, unique set of patient ids in test set
                    svm_models = list, fitted svm models to training set 
                    reduced = boolean, default True, if random sampling of dataset to test of smaller dataset
                    nb_patients = int, size of number of patients selected, ignored if reduced == False
        Returns:
                df_pred = pd.DataFrame, dataframe containing for each patient id the predicted label as a confidence level
    """
    if reduced:
        # sample at random nb_patients patients 
        test_pids = sample(list(test_pids),nb_patients)
        X_test = X_test[test_pids]
    # decision_function returns the distance to the hyperplane 
    y_conf = svm.decision_function(X_test)
    # compute the predictions as confidence levels, ie using sigmoid function instead of sign function
    y_pred = [sigmoid_f(y_conf[i]) for i in range(len(y_conf))]
    # use the mean of the computation for each patient as overall confidence level 
    y_mean = [np.mean(y_pred[i:i+12]) for i in range(len(test_pids))]
    df = pd.DataFrame({sepsis[0]: y_mean},index=test_pids)
    return df


def main(logger):
    """Primary function reading, preprocessing and modelling the data

    Args:
        None

    Returns:
        None
    """
    df_train = pd.read_csv("projects/project_2/data/train_features.csv", nrows=409)
    df_train_label = pd.read_csv("projects/project_2/data/train_labels.csv", nrows=409)
    df_test = pd.read_csv("projects/project_2/data/test_features.csv", nrows=409)

    logger.info('Finished Loading data')


    # list of medical tests that we will have to predict, as well as vital signs (to delete for this task)
    medical_tests = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2"]
    vital_signs = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]
    sepsis = ["LABEL_Sepsis"]

    logger.info('Beginning to deal with missing data')
    # Would be useful to distribute/multithread this part
    # df_train_preprocessed = fill_na_with_average_patient_column(df_train, logger)

    # preprocess testing data
    df_train_preprocessed = fill_na_with_average_column(df_train)
    df_test_preprocessed = fill_na_with_average_column(df_train)
    # df_test_preprocessed = fill_na_with_average_column(df_test)

    # transform training labels for these tasks
    df_train_label[medical_tests+vital_signs+sepsis] = df_train_label[medical_tests+vital_signs+sepsis].astype(int)
    # df_train_label_med = df_train_label.drop(columns=vital_signs+sepsis)
    # df_train_label_sepsis = df_train_label.drop(columns=vital_signs+medical_tests)
    logger.info('Merge labels and features')
    # Merging pids to make sure they map correctly.
    df_train_preprocessed_merged = pd.merge(df_train_preprocessed, df_train_label,  how='left', left_on='pid', right_on ='pid')

    # Transform to arrays
    X_train = df_train_preprocessed_merged.drop(columns=medical_tests+sepsis+vital_signs).values

    # Create list with different label for each medical test
    y_train_set_med = []
    for test in medical_tests:
        y_train_set_med.append(df_train_preprocessed_merged[test].values)
    y_train_sepsis = df_train_preprocessed_merged['LABEL_Sepsis'].values

    # very long to compute
    # compute resampled data for all medical tests
    X_train_resampled_set_med,y_train_resampled_set_med = [0]*len(y_train_set_med),[0]*len(y_train_set_med)

    number_of_tests = len(y_train_set_med)
    for i in range(number_of_tests):

        X_train_resampled_set_med[i], y_train_resampled_set_med[i] = oversampling_strategies(X_train,
                                                                                             y_train_set_med[i],
                                                                                             strategy="adasyn")
        logger.info('Performing oversampling for {} of {} tests.'.format(i, number_of_tests))

    X_train_resampled_sepsis, y_train_resampled_sepsis = oversampling_strategies(X_train, y_train_sepsis,
                                                                                 strategy="adasyn")

    # Modelling

    # For now, use of Linear SVM, scales better to large datasets

    # Linear
    # regularization parameter
    alphas = np.linspace(0.1,10, num=3)
    # to perform either l1 or l2 regularization
    penalty = ["l1", "l2"]

    # for non linear
    # kernel type
    kernels = ["rbf"]
    kernels_backup_tmp = ["linear", "poly", "rbf", "sigmoid"]
    # degree for poly kernel
    degrees = range(1,4)
    # gamma parameter for poly or rbf kernel
    gamma_rbf = np.linspace(0.1,10,num=5)

    # Naïve SVR for all medical tests
    logger.info('Training naive_svm_models.')
    naive_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, medical_tests, alpha=10, typ="naive", reduced=True, size=100, )
    # CV GridSearch with different regularization parameters
    logger.info('Training gridsearch_svm_models.')
    gridsearch_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, medical_tests, param_grid = {"C": alphas, "penalty": penalty}, typ="gridsearch", reduced=False, size=100)
    logger.info('Training naive_non_lin_svm_models.')
    naive_non_lin_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, medical_tests, alpha=10, typ="naive_non_lin", reduced=False, size=100)
    # heavy computation, was too long to run on my machine
    logger.info('Training gridsearch_non_lin_svm_models.')
    gridsearch_non_lin_svm_models = get_models_medical_tests(X_train_resampled_set_med, y_train_resampled_set_med, medical_tests, param_grid = {"C": alphas, "kernel": kernels, "degree": degrees}, typ="gridsearch_non_lin", reduced=False, size=20)
    logger.info('Training naive_sepsis_model.')
    naive_sepsis_model = get_model_sepsis(X_train_resampled_sepsis, y_train_resampled_sepsis, alpha=5,typ="naive", reduced=False, size=100)
    logger.info('Training gridsearch_sepsis_model.')
    gridsearch_sepsis_model = get_model_sepsis(X_train_resampled_sepsis, y_train_resampled_sepsis, param_grid = {"C": alphas, "penalty": penalty}, typ="gridsearch", reduced=False, size=100)
    logger.info('Training non_lin_sepsis_models.')
    non_lin_sepsis_model = get_model_sepsis(X_train_resampled_sepsis, y_train_resampled_sepsis, alpha=5, typ="naive_non_lin", reduced=False, size=100)
    # heavy computation
    logger.info('Training non_lin_gridsearch_sepsis_model.')
    non_lin_gridsearch_sepsis_model = get_model_sepsis(X_train_resampled_sepsis,y_train_resampled_sepsis, param_grid = {"C": alphas, "kernel": kernels, "degree": degrees}, typ="gridsearch_non_lin", reduced=False, size=20)
    X_test = df_test_preprocessed.values
    # get the unique test ids of patients
    test_pids = np.unique(df_test_preprocessed[["pid"]].values)
    naive_predictions = get_predictions(X_test,test_pids,naive_svm_models,medical_tests,reduced=False,nb_patients=100)
    gridsearch_predictions = get_predictions(X_test,test_pids,gridsearch_svm_models,medical_tests,reduced=False,nb_patients=100)
    naive_non_lin_predictions = get_predictions(X_test,test_pids,naive_non_lin_svm_models,medical_tests,reduced=False,nb_patients=100)
    naive_sepsis_predictions = get_sepsis_predictions(X_test,test_pids,naive_sepsis_model,sepsis,reduced=False,nb_patients=100)
    gridsearch_sepsis_predictions = get_sepsis_predictions(X_test,test_pids,gridsearch_sepsis_model,sepsis,reduced=False,nb_patients=100)
    # naive_predictions.head()
    # gridsearch_predictions.head()
    # naive_non_lin_predictions.head()
    # naive_sepsis_predictions.head()
    # gridsearch_sepsis_predictions.head()
    # suppose df is a pandas dataframe containing the result
    # df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

if __name__ == "__main__":
    #
    # parser = argparse.ArgumentParser(
    #     description="CLI args for folder and file \
    # directories"
    # )
    #
    # parser.add_argument(
    #     "--train",
    #     "-tr",
    #     type=str,
    #     required=True,
    #     help="path to the CSV file containing the training \
    #                     data",
    # )
    # parser.add_argument(
    #     "--weights",
    #     "-w",
    #     type=str,
    #     required=True,
    #     help="path where the CSV file where weights should be \
    #                     written",
    # )
    #
    # FLAGS = parser.parse_args()

    # clear logger.
    logging.basicConfig(
        level=logging.DEBUG,
        filename='script_status.log'
    )

    logger = logging.getLogger('Subtask 1 and 2')

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(stream_handler)

    main(logger,)
