#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

from scipy.interpolate import interp1d

"""Implementation of the solution to project 2 from scratch
- For clarity the df_test was renamed to df_val as the test word was used when splitting the 
    labeled data into train and test. Val stands for validation

To try out:
- Preprocessing
    - Tweak data imputer
    - Tweak scaler (Robust scaler, minmax, etc..)
    - Tweak feature selection parameter
    - Tweak order of operations above to see the effect
- Modelling
    - XGBoost
    - SVM

Following conversation with Teammate:
* Rewrite code such that XGBoost is used everywhere
* RandomUnderSampler (without replacement, but should not be a problem)
* Use one line per patient
* Don't do imputation
* Run 150 fits per label

"""


import argparse
import logging
import os
import shutil
import sys
import zipfile
import time
import sys
import torch

import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, r2_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from collections import Counter

# Global variables
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
ESTIMATOR = {"bayesian": BayesianRidge(), "decisiontree": DecisionTreeRegressor(max_features="sqrt", random_state=0), 
                "extratree": ExtraTreesRegressor(n_estimators=10, random_state=0), 
                "knn": KNeighborsRegressor(n_neighbors=10, weights="distance")}

FEATURES_MNAR = ["EtCO2", "PTT", "BUN", "Lactate", "Hgb", "HCO3", "BaseExcess",
                          "Fibrinogen", "Phosphate", "WBC", "Creatinine", "PaCO2", "AST",
                          "FiO2", "Platelets", "SaO2", "Glucose", "Magnesium", "Potassium",
                          "Calcium", "Alkalinephos", "Bilirubin_direct", "Chloride", "Hct",
                          "Bilirubin_total", "TroponinI", "pH"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


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


# ## Load Data

# In[16]:


# df_train = pd.read_csv(r"data/train_features.csv")
df_train_label = pd.read_csv(r"data/train_labels.csv")
# df_val = pd.read_csv(r"data/test_features.csv")

# mnar_columns = [
#         sub + "_presence" for sub in FEATURES_MNAR
#     ]
# pid = df_train["pid"].unique()
# for patient in tqdm(pid):
#     for column in FEATURES_MNAR:
#         presence = int(df_train.loc[
#             df_train["pid"] == patient
#             ][column].any())
#         df_train.at[patient, column] = presence

# print("Done adding features about MNAR features in train")

# # Adding engineered features
# pid = df_val["pid"].unique()
# for patient in tqdm(pid):
#     for column in FEATURES_MNAR:
#         presence = int(df_val.loc[
#             df_val["pid"] == patient
#             ][column].any())
#         df_val.at[patient, column] = presence

# print("Done adding features about MNAR features in val")

# cols_for_timeseries = ["Temp", "RRate", "ABPm", "ABPd", "SpO2", "Heartrate", "ABPs"]
# columns_for_regression = ["Temp", "Hgb", "RRate", "BaseExcess", "WBC", "PaCO2", "FiO2", "Glucose", "ABPm", "ABPd", "SpO2", "Hct", "Heartrate", "ABPs", "pH"]
# columns_for_regression_trend = [
#         sub + "_trend" for sub in columns_for_regression
#     ]
# columns_for_regression_std = [
#         sub + "_std" for sub in columns_for_regression
#     ]
# columns_for_regression_min = [
#         sub + "_min" for sub in columns_for_regression
#     ]
# columns_for_regression_max = [
#         sub + "_max" for sub in columns_for_regression
#     ]
# cols_to_add = columns_for_regression_trend + columns_for_regression_std + columns_for_regression_min + columns_for_regression_max

# df_train = df_train.reindex(
#         df_train.columns.tolist() + cols_to_add,
#         axis=1,
#     )

# train_pid = df_train["pid"].unique()

# for patient in tqdm(train_pid):
#     for i, column in enumerate(columns_for_regression):
#         if df_train.loc[df_train["pid"] == patient][column].isna().sum() <= 8:
#             series = df_train.loc[df_train["pid"] == patient][column]
#             series = series.dropna()
#             standard_deviation = series.std()
#             minimum = series.min()
#             maximum = series.max()
#             X = [i for i in range(0, len(series))]
#             X = np.reshape(X, (len(X), 1))
#             y = series
#             model = LinearRegression()
#             try:
#                 model.fit(X, y)
#                 df_train.at[patient, column + "_trend"] = model.coef_
#             except ValueError:
#                 df_train.at[patient, column + "_trend"] = 0
#             df_train.at[patient, column + "_std"] = standard_deviation
#             df_train.at[patient, column + "_min"] = minimum
#             df_train.at[patient, column + "_max"] = maximum

#     # fill rest of values with 0 for trends col umns
#     df_train[columns_for_regression_trend] = df_train[
#         columns_for_regression_trend
#     ].fillna(value=0)
#     df_train[columns_for_regression_std] = df_train[
#         columns_for_regression_std
#     ].fillna(value=0)
#     df_train[columns_for_regression_min] = df_train[
#         columns_for_regression_min
#     ].fillna(value=0)
#     df_train[columns_for_regression_max] = df_train[
#         columns_for_regression_max
#     ].fillna(value=0)

#     for column in cols_for_timeseries:
#         series = df_train.loc[df_train["pid"] == patient][column]
#         for i, value in enumerate(series):
#             df_train.at[patient, column + f"{i}"] = value

# val_pid = df_val["pid"].unique()

# df_val = df_val.reindex(
#         df_val.columns.tolist() + cols_to_add,
#         axis=1,
#     )

# for patient in tqdm(val_pid):
#     for column in columns_for_regression:
#         if df_val.loc[df_val["pid"] == patient][column].isna().sum() <= 8:
#             series = df_val.loc[df_val["pid"] == patient][column]
#             series = series.dropna()
#             # Drop the rest of the value
#             standard_deviation = series.std()
#             minimum = series.min()
#             maximum = series.max()
#             X = [i for i in range(0, len(series))]
#             X = np.reshape(X, (len(X), 1))
#             y = series
#             model = LinearRegression()
#             try:
#                 model.fit(X, y)
#                 df_val.at[patient, column + "_trend"] = model.coef_
#             except ValueError:
#                 df_val.at[patient, column + "_trend"] = 0
#             df_val.at[patient, column + "_std"] = standard_deviation
#             df_val.at[patient, column + "_min"] = minimum
#             df_val.at[patient, column + "_max"] = maximum

#     # fill rest of values with 0 for trends col umns
#     df_val[columns_for_regression_trend] = df_val[
#         columns_for_regression_trend
#     ].fillna(value=0)
#     df_val[columns_for_regression_std] = df_val[
#         columns_for_regression_std
#     ].fillna(value=0)
#     df_val[columns_for_regression_min] = df_val[
#         columns_for_regression_min
#     ].fillna(value=0)
#     df_val[columns_for_regression_max] = df_val[
#         columns_for_regression_max
#     ].fillna(value=0)

#     for column in cols_for_timeseries:
#         series = df_val.loc[df_val["pid"] == patient][column]
#         for i, value in enumerate(series):
#             df_val.at[patient, column + f"{i}"] = value



# df_train_grouped = pd.DataFrame(index=df_train["pid"].unique(), columns=df_train.columns)

# for patient in tqdm(df_train["pid"].unique()):
#     for column in df_train.columns:
#         patient_timeseries = df_train.loc[df_train["pid"] == patient][column]
#         if patient_timeseries.isnull().all():
#             df_train_grouped.at[patient, column] = np.nan
#         elif column is not "pid":
#             df_train_grouped.at[patient, column] = patient_timeseries.mean()
# df_train = df_train_grouped


# df_val_grouped = pd.DataFrame(index=df_val["pid"].unique(), columns=df_val.columns)

# for patient in tqdm(df_val["pid"].unique()):
#     for column in df_val.columns:
#         patient_timeseries = df_val.loc[df_val["pid"] == patient][column]
#         if patient_timeseries.isnull().all():
#             df_val_grouped.at[patient, column] = np.nan
#         elif column is not "pid":
#             df_val_grouped.at[patient, column] = patient_timeseries.mean()
# df_val = df_val_grouped

# df_train.to_csv("df_train_philip.csv")
# df_val.to_csv("df_val_philip.csv")

df_train_preprocessed = pd.read_csv("df_train_philip.csv")
df_val_preprocessed = pd.read_csv("df_val_philip.csv")

train_pids = np.unique(df_val_preprocessed["pid"].values)
val_pids = np.unique(df_val_preprocessed["pid"].values)

df_train_preprocessed = df_train_preprocessed.sort_values(by=["pid"])
df_train_preprocessed = df_train_preprocessed.drop(columns=IDENTIFIERS)
df_val_preprocessed = df_val_preprocessed.sort_values(by=["pid"])
df_val_preprocessed = df_val_preprocessed.drop(columns=IDENTIFIERS)
df_train_label = df_train_label.sort_values(by=["pid"])


# Data formatting
X_train = df_train_preprocessed.values
X_val = df_val_preprocessed.values
# Create list with different label for each medical test
print("Creating a list of labels for each medical test")
y_train_medical_tests = []
for test in MEDICAL_TESTS:
    y_train_medical_tests.append(df_train_label[test].astype(int).values)

# Create list with different label for sepsis
print("Creating a list of labels for sepsis")
y_train_sepsis = []
for sepsis in SEPSIS:
    y_train_sepsis.append(df_train_label[sepsis].astype(int).values)

# Create list with different label for each vital sign
print("Creating a list of labels for each vital sign")
y_train_vital_signs = []
for sign in VITAL_SIGNS:
    y_train_vital_signs.append(df_train_label[sign].astype(int).values)

# Scale data 
# scaler = StandardScaler(with_mean=True, with_std=True)
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
X_train_scaled = X_train
X_val_scaled = X_val
# ## Modelling medical tests
# Modelling using extreme gradient boosting
clf = xgb.XGBClassifier(objective="binary:logistic", n_thread=-1)
models = []
losses = []
feature_selector_medical_tests = []

for i, test in enumerate(MEDICAL_TESTS):
    print(f"Fitting model for {test}.")
    X_train, X_test, y_train, y_test = train_test_split(
    X_train_scaled, y_train_medical_tests[i], test_size=0.10, random_state=42, shuffle=True
    )

    print("Downsampling")
    sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    # scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1]

    print("Fitting coarse model")
    # Coarse parameter grid not optimized at all yet
    coarse_param_grid = {
        "booster": ["dart"],
        "eta": np.arange(0,1,0.1),
        "min_child_weight": range(1, 10, 1),
        "max_depth": range(4, 10, 1),
        "gamma": range(0, 100, 1),
        "max_delta_step": range(1, 10, 1),
        "subsample": np.arange(0.1, 1, 0.05),
        "colsample_bytree": np.arange(0.3, 1, 0.05),
        "n_estimators": range(50, 150, 1),
        "scale_pos_weight": [1],
        "reg_lambda": [0, 1], # Ridge regularization
        "reg_alpha": [0, 1], # Lasso regularization
        "eval_metric": ["error"],
        "verbosity": [1]
    }
    coarse_search = RandomizedSearchCV(estimator=clf,
            param_distributions=coarse_param_grid, scoring="roc_auc",
            n_jobs=-1, cv=10, n_iter=150, verbose=1)
    coarse_search.fit(X_train, y_train)
    print(coarse_search.best_estimator_.predict_proba(X_test)[:,1])
    print(f"ROC score on test set {roc_auc_score(y_test, coarse_search.best_estimator_.predict_proba(X_test)[:,1])}")
    print(f"CV score {coarse_search.best_score_}")
    best_params = coarse_search.best_params_
    print(f"ROC score on test set {roc_auc_score(y_test, coarse_search.best_estimator_.predict_proba(X_test)[:,1])}")
    print(f"CV score {coarse_search.best_score_}")
    
    
    
    models.append(coarse_search.best_estimator_)
    
print(f"Finished test for medical tests.")


# In[102]:


import joblib
for i, model in enumerate(models):
    joblib.dump(models[i], f"xgboost_fine_{MEDICAL_TESTS[i]}.pkl")


# In[105]:


# Get predictions for medical tests
df_pred_medical_test = pd.DataFrame(index=val_pids, columns=MEDICAL_TESTS)
for i, test in enumerate(MEDICAL_TESTS):
    model_for_test = models[i]
#     print(model_for_test.predict_proba(X_val_vital_sign))
    y_pred = model_for_test.predict_proba(X_val_scaled)[:, 1]
    df_pred_medical_test[test] = y_pred

df_pred_medical_test = df_pred_medical_test.reset_index().rename(columns={"index": "pid"})


# ## Modelling sepsis

# In[107]:


# Model and predict sepsis

clf = xgb.XGBClassifier(objective="binary:logistic", n_thread=-1)


X_train, X_test, y_train, y_test = train_test_split(
    X_train_scaled, y_train_sepsis[0], test_size=0.10, random_state=42, shuffle=True
)

# scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1]
print("Downsampling")
sampler = RandomUnderSampler(random_state=42)
X_train, y_train = sampler.fit_resample(X_train, y_train)

param_grid = {
        "booster": ["dart"],
        "eta": np.arange(0,1,0.1),
        "min_child_weight": range(1, 10, 1),
        "max_depth": range(4, 10, 1),
        "gamma": range(0, 100, 1),
        "max_delta_step": range(1, 10, 1),
        "subsample": np.arange(0.1, 1, 0.05),
        "colsample_bytree": np.arange(0.3, 1, 0.05),
        "n_estimators": range(50, 150, 1),
        "scale_pos_weight": [1],
        "reg_lambda": [0, 1], # Ridge regularization
        "reg_alpha": [0, 1], # Lasso regularization
        "eval_metric": ["error"],
        "verbosity": [1]
    }


print("Fitting model")
coarse_search = RandomizedSearchCV(estimator=clf,
        param_distributions=param_grid, scoring="roc_auc",
        n_jobs=-1, cv=10, n_iter=150, verbose=1)
coarse_search.fit(X_train, y_train)

sepsis_model = coarse_search.best_estimator_
print(f"ROC score on test set {roc_auc_score(y_test, coarse_search.best_estimator_.predict_proba(X_test)[:,1])}")
print(f"CV score {coarse_search.best_score_}")
print(f"Finished test for medical tests.")


# In[110]:


y_pred = sepsis_model.predict_proba(X_val_scaled)[:,1]
df_pred_sepsis = pd.DataFrame(y_pred, index=val_pids, columns=SEPSIS)
df_pred_sepsis = df_pred_sepsis.reset_index().rename(columns={"index": "pid"})


# In[111]:


joblib.dump(sepsis_model, f"xgboost_fine_sepsis.pkl")


# ## Modelling vital signs



# Modelling of vital signs
models = []
losses = []
feature_selectors_vital_signs = []
clf = xgb.XGBRegressor(objective="reg:squarederror", n_thread=-1)

for i, sign in enumerate(VITAL_SIGNS):
    print(f"Fitting model for {sign}.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled, y_train_vital_signs[i], test_size=0.10, random_state=42, shuffle=True
    )

    print("Fitting model")
    
    param_grid = {
        "booster": ["dart"],
        "eta": np.arange(0,1,0.1),
        "min_child_weight": range(1, 10, 1),
        "max_depth": range(4, 10, 1),
        "gamma": range(0, 100, 1),
        "max_delta_step": range(1, 10, 1),
        "subsample": np.arange(0.1, 1, 0.05),
        "colsample_bytree": np.arange(0.3, 1, 0.05),
        "n_estimators": range(50, 150, 1),
        "scale_pos_weight": [1],
        "reg_lambda": [0, 1], # Ridge regularization
        "reg_alpha": [0, 1], # Lasso regularization
        "eval_metric": ["error"],
        "verbosity": [1]
    }


    
    coarse_search = RandomizedSearchCV(estimator=clf,
            param_distributions=param_grid, scoring="r2",
            n_jobs=-1, cv=10, n_iter=150, verbose=1)
    coarse_search.fit(X_train, y_train)
    models.append(coarse_search.best_estimator_)
    print(f"CV score {coarse_search.best_score_}")
    print(f"Test score is {r2_score(y_test, coarse_search.best_estimator_.predict(X_test))}")
    print(f"Finished test for medical tests.")


for i, model in enumerate(models):
    joblib.dump(models[i], f"xgboost_fine_{VITAL_SIGNS[i]}.pkl")


# Get predictions for vital signs using ANN
df_pred_vital_signs = pd.DataFrame(index=val_pids, columns=VITAL_SIGNS)
for i, vital_sign in enumerate(VITAL_SIGNS):
    model = models[i]
    y_pred = model.predict(X_val_scaled)
    df_pred_vital_signs[vital_sign] = y_pred

df_pred_vital_signs = df_pred_vital_signs.reset_index().rename(columns={"index": "pid"})

# ## Export to ZIP file

df_predictions = pd.merge(df_pred_medical_test, df_pred_sepsis, on="pid")
df_predictions = pd.merge(df_predictions, df_pred_vital_signs, on="pid")
print("Export predictions DataFrame to a zip file")
print(df_predictions)
df_predictions.to_csv(
    "predictions.csv",
    index=None,
    sep=",",
    header=True,
    encoding="utf-8-sig",
    float_format="%.2f",
)

with zipfile.ZipFile("predictions.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("predictions.csv")
os.remove("predictions.csv")

