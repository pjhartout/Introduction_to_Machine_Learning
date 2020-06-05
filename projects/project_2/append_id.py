#!/usr/bin/env python

# This is a short script to append the patient ids to the predictions

import pandas as pd
import numpy as np

predictions = pd.read_csv("predictions.csv")

val_features = pd.read_csv("data/test_features.csv")

sorted_pid = np.sort(val_features["pid"].unique())

predictions.insert(0, 'pid', sorted_pid)

predictions.to_csv("predictions.csv")
