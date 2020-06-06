#!/usr/bin/env python
# coding: utf-8


"""project_4.py

This is a rework of project 4 with a generator and ways to optimize memory
"""

__author__ = "Philip Hartout; Josephine Yates"
__email__ = "phartout@student.ethz.ch; jyates@student.ethz.ch"

import keras
import tensorflow as tf
import tensorflow_addons as tfa

import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl
from keras.preprocessing.image import img_to_array
from keras import optimizers
from itertools import chain

import sys
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


T_G_WIDTH = 224
T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
CHUNK_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
USE_PRETRAINED_MODEL = True
EMBEDDING_SIZE = 300
EPOCHS = 100
MIN_EPOCHS = 10  # minimal number of epochs before early stopping takes effect.
STEPS_PER_EPOCH = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE

###############################################################################
# Data preprocessing functions
###############################################################################


def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT, T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.vgg16.preprocess_input(
        t_image, data_format="channels_last"
    )
    return t_image


def image_generator(df_to_load):
    """Data generator loading images into memory for one batch

    Args:
        files_to_load (pd.core.DataFrame): dataframe loaded into memory

    Yields:
        anchor, positive and negative samples
    """

    # load file names
    list_of_files = list(chain(*df_to_load.values.tolist()))
    files_to_load = [str(file) + ".jpg" for file in list_of_files]

    img_array = {}
    for file in files_to_load:
        img = t_read_image(os.path.join("data/food", file))
        img_array[file.split(".jpg")[0]] = img_to_array(img)

    anchors_train = [img_array[img] for img in np.array(df_to_load["A"])]
    positives_train = [img_array[img] for img in np.array(df_to_load["B"])]
    negatives_train = [img_array[img] for img in np.array(df_to_load["C"])]

    y = np.random.randint(2, size=(1, 2, len(anchors_train))).T

    yield ([anchors_train, positives_train, negatives_train], y)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


###############################################################################
# Model definition functions
###############################################################################


def createNASNetLargeModel(emb_size):
    # Initialize a NASNetLargeModel model
    nasnetlarge_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    nasnetlarge_model = keras.applications.NASNetLarge(
        include_top=False,
        weights="imagenet",
        input_tensor=nasnetlarge_input,
        input_shape=None,
        pooling=None,
    )

    # New Layers over InceptionResNetV2
    net = nasnetlarge_model.output
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(net)

    # model creation
    base_model = Model(nasnetlarge_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape = (T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name="input_anchor")
    input_positive = kl.Input(shape=input_shape, name="input_pos")
    input_negative = kl.Input(shape=input_shape, name="input_neg")
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name="pos_dist")(
        [net_anchor, net_positive]
    )
    negative_dist = kl.Lambda(euclidean_distance, name="neg_dist")(
        [net_anchor, net_negative]
    )
    tertiary_dist = kl.Lambda(euclidean_distance, name="ter_dist")(
        [net_positive, net_negative]
    )

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = kl.Lambda(
        lambda vects: K.stack(vects, axis=1), name="stacked_dists"
    )([positive_dist, negative_dist, tertiary_dist])

    model = Model(
        [input_anchor, input_positive, input_negative],
        stacked_dists,
        name="triple_siamese",
    )

    v_optimizer = optimizers.Adam(lr=LEARNING_RATE)

    model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

    return model


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(
        K.maximum(
            K.constant(0),
            K.square(y_pred[:, 0, 0])
            - 0.5 * (K.square(y_pred[:, 1, 0]) + K.square(y_pred[:, 2, 0]))
            + margin,
        )
    )


def accuracy(y_true, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])


def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


###############################################################################
# Preprocessing execution
###############################################################################

print("Reading file lists")
path, dirs, files = next(os.walk("data/food"))

train_triplets = pd.read_csv("data/train_triplets.txt", names=["A", "B", "C"], sep=" ")
test_triplets = pd.read_csv("data/test_triplets.txt", names=["A", "B", "C"], sep=" ")

for column in train_triplets.columns:
    train_triplets[column] = train_triplets[column].astype(str)
    test_triplets[column] = test_triplets[column].astype(str)
    train_triplets[column] = train_triplets[column].apply(lambda x: x.zfill(5))
    test_triplets[column] = test_triplets[column].apply(lambda x: x.zfill(5))


print("Creating dataset generator")
data_generator = image_generator(train_triplets)

print("Creating dataset object")
train_data = tf.data.Dataset.from_generator(
    lambda: data_generator,
    output_types=(tf.float32, tf.float32, tf.float32, tf.int8),
    output_shapes=(None, None, None, []),
)

print("Transform to dataset instance")
train_data = prepare_for_training(train_data, cache=True, shuffle_buffer_size=1000)

print("Prepare print")
print(list(train_data.prefetch(2).as_numpy_iterator()))

###############################################################################
# Model training and validation
###############################################################################

# model = createNASNetLargeModel(EMBEDDING_SIZE)

# callbacks = [
#     tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath="model.{epoch:02d}-{val_loss:.2f}.h5"
#     ),
# ]

# model.fit(
#     train_data,
#     validation_split=0.1,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     callbacks=callbacks,
#     workers=20,
#     use_multiprocessing=True,
#     max_queue_size=10
# )

###############################################################################
# Model inference
###############################################################################
