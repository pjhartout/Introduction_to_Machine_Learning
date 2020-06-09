#!/usr/bin/env python
# coding: utf-8

__author__ = "Philip Hartout; Josephine Yates"
__email__ = "phartout@student.ethz.ch; jyates@student.ethz.ch"


"""project_4_trial_4.py
This is a rework of project 4 with a generator and ways to optimize memory

Notice: in its current form the code does not run. the
tf.data.Dataset.from_generator yields two numpy arrays: one of size (n,2,1) and
one of size (3, n, T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS) which is what is
yielded by the tensorflow dataset object from_generator instead of a list.

Feed multiple inputs to keras model is possible:
https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras

"""

import cv2
import os

import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Lambda

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

import tensorflow as tf
import numpy as np
import pandas as pd


T_G_SEED = 42
tf.random.set_seed(T_G_SEED)
np.random.seed(T_G_SEED)
T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS = 50, 50, 3
CHUNK_SIZE = 100
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 300
EPOCHS = 100
STEPS_PER_EPOCH = 10
VAL_FRAC = 0.1
AUTOTUNE = tf.data.experimental.AUTOTUNE
LOAD_ROWS = None
CHUNK_SIZE = 2
TRAIN_TRIPLETS = "data/train_triplets.txt"
TEST_TRIPLETS = "data/test_triplets.txt"
###############################################################################
# Model definition functions
###############################################################################


def triplet_loss(y_true, y_pred):

    # y_true is passed by keras but unused in this context, therefore it can be
    # set to random earlier.
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
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())
    )


def createModel(emb_size):
    # This function needs to be refined to take in one array of size
    # (3, n, T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS) which is what is yielded by
    # the tensorflow dataset object from_generator instead of a list.

    # Initialize a ResNet50_ImageNet Model
    resnet_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    resnet_model = tf.keras.applications.resnet50.ResNet50(
        weights="imagenet", include_top=False, input_tensor=resnet_input
    )

    # New Layers over ResNet50
    net = resnet_model.output
    # net = kl.Flatten(name='flatten')(net)
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    # net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(
        net
    )

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape = (T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name="input_anchor")
    input_positive = kl.Input(shape=input_shape, name="input_pos")
    input_negative = kl.Input(shape=input_shape, name="input_neg")

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function.
    # Here it's Euclidean distance.

    positive_dist = kl.Lambda(euclidean_distance, name="pos_dist")(
        [net_anchor, net_positive]
    )
    negative_dist = kl.Lambda(euclidean_distance, name="neg_dist")(
        [net_anchor, net_negative]
    )
    tertiary_dist = kl.Lambda(euclidean_distance, name="ter_dist")(
        [net_positive, net_negative]
    )

    # This lambda layer simply stacks outputs so both distances are available
    # to the objective

    stacked_dists = kl.Lambda(
        lambda vects: K.stack(vects, axis=1), name="stacked_dists"
    )([positive_dist, negative_dist, tertiary_dist])

    model = Model(
        [input_anchor, input_positive, input_negative],
        stacked_dists,
        name="triple_siamese",
    )

    # Setting up optimizer designed for variable learning rate

    # Variable Learning Rate per Layers
    lr_mult_dict = {}

    for layer in resnet_model.layers:
        # comment this out to refine earlier layers
        layer.trainable = False
        print(layer.name)
        lr_mult_dict[layer.name] = 1

    v_optimizer = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

    return model


###############################################################################
# Data preprocessing functions
###############################################################################


def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT, T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = tf.keras.applications.resnet50.preprocess_input(
        t_image, data_format="channels_last"
    )
    return t_image


DATA_GEN_ARGS = {}

datagen = ImageDataGenerator(**DATA_GEN_ARGS)


def create_data_generator(X1, X2, X3, Y):
    local_seed = T_G_SEED
    genX1 = datagen.flow(
        X1, Y, batch_size=len(X1), seed=local_seed, shuffle=False
    )
    genX2 = datagen.flow(
        X2, Y, batch_size=len(X2), seed=local_seed, shuffle=False
    )
    genX3 = datagen.flow(
        X3, Y, batch_size=len(X3), seed=local_seed, shuffle=False
    )
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()

        yield [X1i[0], X2i[0], X3i[0]], X1i[1]


###############################################################################
# Data preprocessing functions
###############################################################################


def main():
    train_triplets = pd.read_csv(
        TRAIN_TRIPLETS,
        names=["A", "B", "C"],
        sep=" ",
        nrows=LOAD_ROWS,
        dtype=str,
    )

    val_triplets = train_triplets.sample(frac=VAL_FRAC)
    train_triplets = train_triplets.drop(val_triplets.index)

    val_triplets_len = len(val_triplets)
    train_triplets_len = len(train_triplets)

    total_train_chunks = int(np.ceil(train_triplets_len / float(CHUNK_SIZE)))
    total_val_chunks = int(np.ceil(val_triplets_len / float(CHUNK_SIZE)))

    # load file names
    for main_dir, subdir, file in os.walk(r"data/food/"):
        list_dir = file[:]
        # load the image
    img_array = {}
    for file in tqdm(list_dir):
        img = t_read_image(os.path.join("data/food", file))
        img_array[file.split(".jpg")[0]] = img_to_array(img)

    print("Getting anchors train ...")
    anchors_train = [img_array[img] for img in np.array(train_triplets["A"])]
    print("Getting positives train ...")
    positives_train = [img_array[img] for img in np.array(train_triplets["B"])]
    print("Getting negatives train ...")
    negatives_train = [img_array[img] for img in np.array(train_triplets["C"])]

    print("Creating model ...")
    model = createModel(EMBEDDING_SIZE)
    for e in range(EPOCHS):
        for chunk in range(total_train_chunks):
            anchors_train_chunk = np.array(
                anchors_train[chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE]
            )

            positives_train_chunk = np.array(
                positives_train[chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE]
            )

            negatives_train_chunk = np.array(
                negatives_train[chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE]
            )

            y = np.random.randint(2, size=(1, 2, len(anchors_train_chunk))).T

            gen = create_data_generator(
                anchors_train_chunk,
                positives_train_chunk,
                negatives_train_chunk,
                y,
            )
            training_dataset = tf.data.Dataset.from_generator(
                lambda: gen, output_types=(tf.float32, tf.float32)
            )
            # training_dataset.prefetch(AUTOTUNE)
            model.fit(
                training_dataset,
                steps_per_epoch=len(y) / BATCH_SIZE,
                epochs=1,
                shuffle=False,
                use_multiprocessing=True,
            )


if __name__ == "__main__":
    main()
