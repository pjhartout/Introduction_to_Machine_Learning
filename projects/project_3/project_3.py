#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tf.keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras import optimizers

import csv
import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

df_train = pd.read_csv("data\\train.csv")
df_val = pd.read_csv("data\\test.csv")

# very imbalanced!
df_train.describe()

# split strings into amino acids sequences
X_train = df_train["Sequence"].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
y_train = df_train["Active"].values
X_val = df_val["Sequence"].values
X_val = [list(X_val[i]) for i in range(len(X_val))]

# percentage active
print("Percentage active mutations : ",np.around(sum(y_train)/len(y_train)*100,2)," %")

# one hot encode the mutations, taking into consideration mutation position
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
X_train_onehot = enc.transform(X_train).toarray()
X_val_onehot = enc.transform(X_val).toarray()

# scale the input
#scaler = StandardScaler()
scaler = PowerTransformer()
X_train_scaled =scaler.fit_transform(X_train_onehot)
X_val_scaled = scaler.transform(X_val_onehot)

# define f1 score, precision and recall for keras to be able to follow real time
# taken from https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
# and https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# construct ANN to perform binary classification
def get_ANN(X_train_scaled,y_train,n_layers,hidden_units):
    """Constructs an ANN model to perform binary classification

    Args: X_train_scaled (np.ndarray): scaled array of one hot encoded AA mutation sequences
        y_train (np.ndarray): labels (active or inactive)
        n_layers (int): number of hidden layers for the ANN
        hidden_units (int): number of units per hidden layer

    Returns: model (keras.models.Sequential): trained ANN
    """
    print("Starting train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=0.15, random_state=42, shuffle=True
    )

    print("Resampling to account for imbalance in data")
    sampler = ADASYN()
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    # ANN architecture definition
    model = Sequential()
    model.add(Dense(hidden_units, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    for i in range(1,n_layers):
        model.add(Dense(hidden_units, activation="relu"))
        model.add(Dropout(0.5))
    model.add(Dense(1,activation="sigmoid"))

    # use Adam as optimizer
    opt = Adam()

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[precision_m, recall_m, get_f1])
    model.fit(X_train_res,y_train_res,epochs=65,batch_size=32)

    # evaluate the model on test set
    score = model.evaluate(X_test, y_test, batch_size=64)
    print("Loss, precision, recall, F1 : ",score)

    return model

# train the ANN
## beware the real time loss, precision, recall and F1 are calculated on batches so are not accurate
model = get_ANN(X_train_scaled, y_train, 3, 125)


# perform predictions
y_pred = np.around(model.predict(X_val_scaled))


# save to csv
np.savetxt("predictions.csv", y_pred, fmt="%i")
