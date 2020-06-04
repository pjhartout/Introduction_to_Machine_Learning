#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

""" This is the Implementation of project 3, version 3.
This script is meant to be executed from the root of the IML repository.

"""

import joblib
import zipfile
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PERCENT_PRESENT_THRESHOLD = (
    0.8
)  # columns containing >PERCENT_PRESENT_THRESHOLD will be unstacked
N_ITER = 100  # Number of models to be fitted for each
CV_FOLDS = 10 # Cross validation folds

PARAM_DIST = {
        "n_estimators": stats.randint(150, 500),
        "learning_rate": stats.uniform(0.01, 0.07),
        "max_depth": [4, 5, 6, 7],
        "colsample_bytree": stats.uniform(0.5, 0.45),
        "min_child_weight": [1, 2, 3],
}

IDENTIFIERS = ["pid", "Time"]

CLASSIFIERS = [
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
    "LABEL_Sepsis",
]

REGRESSORS = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]

SEPSIS = ["LABEL_Sepsis"]

print("Loading data.")
df_train_features = pd.read_csv("projects/project_2/data/train_features.csv")
df_train_labels = pd.read_csv("projects/project_2/data/train_labels.csv")
df_test_features = pd.read_csv("projects/project_2/data/test_features.csv")
print("Data loaded.")
####################################################################################################
# Set and sort indices
####################################################################################################
df_train_labels = df_train_labels.set_index("pid")
df_train_labels = df_train_labels.sort_index()
df_train_features = df_train_features.set_index(IDENTIFIERS)
df_train_features = df_train_features.sort_index()
df_test_features = df_test_features.set_index(IDENTIFIERS)
df_test_features = df_test_features.sort_index()

####################################################################################################
# Preprocessing training data
####################################################################################################

percent_missing = df_train_features.isnull().sum() * 100 / len(df_train_features)

features_to_time_series = percent_missing[
    (percent_missing > 0) & (percent_missing < (1 - PERCENT_PRESENT_THRESHOLD) * 100)
    ].index.tolist()

# This resets the time index to indicate that all values have the same secondary index (0-11)
df_train_features.index = pd.MultiIndex.from_arrays(
    [
        df_train_features.index.get_level_values(0),
        df_train_features.groupby(level=0).cumcount(),
    ],
    names=["pid", "Time"],
)
# Unstack columns which have multiple values available as determined in percent_missing
df_train_features_unstacked = df_train_features[features_to_time_series].unstack(
    level=-1
)

# Take the median of all other columns
features_to_median = np.setdiff1d(df_train_features.columns, features_to_time_series)
df_train_features_median = (
    df_train_features[features_to_median].groupby(level=0).median()
)

mask_column_names = [sub + "_presence" for sub in features_to_median]

# Add masking variable
df_train_features_mask = df_train_features_median.mask(
    df_train_features_median.isna(), 0
)
df_train_features_mask[df_train_features_mask != 0] = 1

df_train_features_median = df_train_features_median.join(
    df_train_features_mask, rsuffix="_presence"
)

# Merge result
df_train_features = pd.merge(
    df_train_features_unstacked, df_train_features_median, on="pid"
)

# Take care of interpolation on time series data
print("Interpolate time series training features")
for i in tqdm(range(len(features_to_time_series))):
    df_train_features[df_train_features.columns[i: i + 12]] = df_train_features[
        df_train_features.columns[i: i + 12]
    ].interpolate(axis=1)

df_train_labels.join(df_train_features)

scaler = StandardScaler()
X_train_preprocessed = scaler.fit_transform(df_train_features)

####################################################################################################
# Preprocessing testing data
####################################################################################################

# This resets the time index to indicate that all values have the same secondary index (0-11)
df_test_features.index = pd.MultiIndex.from_arrays(
    [
        df_test_features.index.get_level_values(0),
        df_test_features.groupby(level=0).cumcount(),
    ],
    names=["pid", "Time"],
)
# Unstack columns which have multiple values available as determined in percent_missing
df_test_features_unstacked = df_test_features[features_to_time_series].unstack(level=-1)

# Take the median of all other columns
df_test_features_median = df_test_features[features_to_median].groupby(level=0).median()

# Add masking variable
df_test_features_mask = df_test_features_median.mask(df_test_features_median.isna(), 0)
df_test_features_mask[df_test_features_mask != 0] = 1

df_test_features_median = df_test_features_median.join(
    df_test_features_mask, rsuffix="_presence"
)

# Merge result
df_test_features = pd.merge(
    df_test_features_unstacked, df_test_features_median, on="pid"
)

# Take care of interpolation on time series data
print("Interpolate time series testing features")
for i in tqdm(range(len(features_to_time_series))):
    df_test_features[df_test_features.columns[i: i + 12]] = df_test_features[
        df_test_features.columns[i: i + 12]
    ].interpolate(axis=1)

# Scale the data using tranform
X_val = scaler.transform(df_test_features)

print("Finished feature preprocessing.")

####################################################################################################
# Preprocessing prediction df
####################################################################################################

val_pids = np.unique(df_test_features.index.values)
df_pred_clf = pd.DataFrame(index=val_pids, columns=CLASSIFIERS)
df_pred_reg = pd.DataFrame(index=val_pids, columns=REGRESSORS)
print("Finished prediction df preparations.")

####################################################################################################
# Training all classifiers
####################################################################################################
print("Starting training models medical tests")
for i, clf in enumerate(CLASSIFIERS):
    print(f"Fitting model for {clf}.")
    y_train = df_train_labels[clf].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_preprocessed, y_train, test_size=0.10, random_state=42, shuffle=True
    )

    print("Downsampling")
    sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(objective="binary:logistic", n_thread=-1)

    clf_search = RandomizedSearchCV(
        model,
        param_distributions=PARAM_DIST,
        cv=CV_FOLDS,
        n_iter=N_ITER,
        scoring="roc_auc",
        error_score=0,
        verbose=3,
        n_jobs=-1,
    )

    clf_search.fit(X_train, y_train)

    # Performance outline
    print(clf_search.best_estimator_.predict_proba(X_test)[:, 1])
    print(
        f"ROC score on test set "
        f"{roc_auc_score(y_test, clf_search.best_estimator_.predict_proba(X_test)[:, 1])}"
    )
    print(f"CV score {clf_search.best_score_}")
    print(f"Best parameters {clf_search.best_params_}")

    # Model persistence
    joblib.dump(
        clf_search.best_estimator_,
        f"projects/project_2/xgboost_fine_{CLASSIFIERS[i]}.pkl",
    )

    # Validation predictions
    y_pred = clf_search.best_estimator_.predict_proba(X_val)[:, 1]
    df_pred_clf[clf] = y_pred

print("Finished training classifiers")

####################################################################################################
# Training all regressors
####################################################################################################
for i, reg in enumerate(REGRESSORS):
    print(f"Fitting model for {reg}.")
    y_train = df_train_labels[reg].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_preprocessed, y_train, test_size=0.10, random_state=42, shuffle=True
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", n_thread=-1)

    regressor_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DIST,
        scoring="r2",
        n_jobs=-1,
        cv=CV_FOLDS,
        n_iter=N_ITER,
        verbose=1,
    )

    regressor_search.fit(X_train, y_train)
    print(f"CV score {regressor_search.best_score_}")
    print(
        f"Test score is {r2_score(y_test, regressor_search.best_estimator_.predict(X_test))}"
    )
    print(f"Finished test for medical tests.")

    # Model persistence
    joblib.dump(
        regressor_search.best_estimator_,
        f"projects/project_2/xgboost_fine_{CLASSIFIERS[i]}.pkl",
    )

    y_pred = regressor_search.best_estimator_.predict(X_val)
    df_pred_reg[reg] = y_pred

####################################################################################################
# Process and export predictions
####################################################################################################
df_predictions = df_pred_clf.join(df_pred_reg).sort_index()

print("Export predictions DataFrame to a zip file")
df_predictions.to_csv(
    "projects/project_2/predictions.csv",
    index=None,
    sep=",",
    header=True,
    encoding="utf-8-sig",
    float_format="%.2f",
)

with zipfile.ZipFile(
        "projects/project_2/predictions.zip", "w", compression=zipfile.ZIP_DEFLATED
) as zf:
    zf.write("projects/project_2/predictions.csv")
os.remove("projects/project_2/predictions.csv")
