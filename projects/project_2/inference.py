#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

""" This script generates predictions for project 2 based on the persistent
models stored as .pkl files.

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
from glob import iglob
from itertools import chain

rootdir = (
    "/home/pjh/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/"
)

PERCENT_PRESENT_THRESHOLD = (
    0.8
)  # columns containing >PERCENT_PRESENT_THRESHOLD will be unstacked

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
    df_train_features[df_train_features.columns[i : i + 12]] = df_train_features[
        df_train_features.columns[i : i + 12]
    ].interpolate(axis=1)

df_train_labels.join(df_train_features)

scaler = StandardScaler()
X_train_preprocessed = scaler.fit_transform(df_train_features)

###############################################################################
# Preprocessing testing data
###############################################################################

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
    df_test_features[df_test_features.columns[i : i + 12]] = df_test_features[
        df_test_features.columns[i : i + 12]
    ].interpolate(axis=1)

# Scale the data using tranform
X_val = scaler.transform(df_test_features)

print("Finished feature preprocessing.")

###############################################################################
# Preprocessing prediction df
###############################################################################

val_pids = np.unique(df_test_features.index.values)
df_pred = pd.DataFrame(index=val_pids, columns=CLASSIFIERS+REGRESSORS)
print("Finished prediction df preparations.")
print(f"Classifiers: {CLASSIFIERS}")
print(f"Regressors: {REGRESSORS}")
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith((".pkl")):
            model = joblib.load(os.path.join(subdir, file))
            filename = os.path.splitext(os.path.basename(file))[0].replace(
                "xgboost_fine_", ""
            )
            print(filename)
            print(any(filename in clf for clf in CLASSIFIERS))
#             if any(filename in clf for clf in CLASSIFIERS):
#                 print(f"Predicting {filename} as a regressor")
#                 prediction = model.predict_proba(X_val)[:, 1]
#             else:
#                 print(f"Predicting {filename} as a regressor")
#                 prediction = model.predict(X_val)
#             df_pred[filename] = prediction

# print("Export predictions DataFrame to a zip file")
# df_pred.to_csv(
#     "projects/project_2/predictions.csv",
#     index=None,
#     sep=",",
#     header=True,
#     encoding="utf-8-sig",
#     float_format="%.2f",
# )

# with zipfile.ZipFile(
#         "projects/project_2/predictions.zip", "w", compression=zipfile.ZIP_DEFLATED
# ) as zf:
#     zf.write("predictions.csv")
# os.remove("projects/project_2/predictions.csv")
