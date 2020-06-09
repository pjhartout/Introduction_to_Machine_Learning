#!/usr/bin/env python
# coding: utf-8


"""project_4_trial_3.py DEPRECATED - REWORK FROM BAS
This is a rework of project 4 with a generator and ways to optimize memory
This was reworked from project_4_trial_2.py by Bas.
"""

__author__ = "Philip Hartout; Josephine Yates"
__email__ = "phartout@student.ethz.ch; jyates@student.ethz.ch"

import cv2, os, sys

from itertools import chain

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    img_to_array,
    ImageDataGenerator,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
)
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import pandas as pd

T_G_SEED = 42
tf.set_random_seed(T_G_SEED)
np.random.seed(T_G_SEED)
T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS = 331, 331, 3
CHUNK_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 300
EPOCHS = 100
STEPS_PER_EPOCH = 10
TEST_SAMPLES = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE
LOAD_ROWS = None
IMG_DIR = "/home/bts/bas/"
TRAIN_TRIPLETS = "/home/bts/bas/train_triplets.txt"
TEST_TRIPLETS = "/home/bts/bas/test_triplets.txt"

datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)


def t_read_image(loc):
    t_image = cv2.imread(loc).astype("float32")
    t_image = cv2.resize(t_image, (T_G_HEIGHT, T_G_WIDTH))
    t_image = preprocess_input(t_image, data_format="channels_last")
    return t_image


def image_generator(df_to_load):
    """Data generator loading images into memory for one batch

    Args:
        df_to_load (pd.core.DataFrame): dataframe loaded into memory

    Yields:
        inputs (dict): ?
        outputs (dict): ?
    """
    # load file names
    list_of_files = list(chain(*df_to_load.values.tolist()))
    files_to_load = [str(f) + ".jpg" for f in list_of_files]

    img_array = {}
    # Use f instead of file, since file is a reserved key word
    for f in files_to_load:
        img = t_read_image(os.path.join(IMG_DIR, f))
        img_array[f.split(".jpg")[0]] = img_to_array(img)

    frame_size = BATCH_SIZE

    while not df_to_load.empty:
        if frame_size > len(df_to_load):
            frame_size = len(df_to_load)

        batch_images = df_to_load.sample(
            n=BATCH_SIZE, replace=False, random_state=42
        )

        train_triplets = df_to_load.drop(batch_images.index)

        anchor_imgs = np.array(batch_images["A"])
        pos_imgs = np.array(batch_images["B"])
        neg_imgs = np.array(batch_images["C"])

        anchors_train = [img_array[img] for img in anchor_imgs]
        positives_train = [img_array[img] for img in pos_imgs]
        negatives_train = [img_array[img] for img in neg_imgs]
        # What is happening here?
        y = np.random.randint(2, size=(1, 2, len(anchors_train))).T
        inputs = (anchors_train, positives_train, negatives_train)
        outputs = y

        yield inputs, outputs


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
    nasnet = NASNetLarge(
        include_top=False,
        weights="imagenet",
        input_shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS),
        pooling=None,
    )

    # New Layers over InceptionResNetV2 -- NASNetLarge?
    inputs = nasnet.input
    outputs = nasnet.output
    outputs = GlobalAveragePooling2D(name="gap")(outputs)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(emb_size, activation="relu", name="t_emb_1")(outputs)
    outputs = Lambda(lambda x: l2_normalize(x, axis=1), name="t_emb_1_l2norm")(
        outputs
    )

    base_model = Model(inputs, outputs, name="base_model")

    # triplet framework, shared weights
    input_shape = (T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS)
    input_anchor = Input(shape=input_shape, name="input_anchor")
    input_positive = Input(shape=input_shape, name="input_pos")
    input_negative = Input(shape=input_shape, name="input_neg")
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    positive_dist = Lambda(euclidean_distance, name="pos_dist")(
        [net_anchor, net_positive]
    )
    negative_dist = Lambda(euclidean_distance, name="neg_dist")(
        [net_anchor, net_negative]
    )
    tertiary_dist = Lambda(euclidean_distance, name="ter_dist")(
        [net_positive, net_negative]
    )

    # This lambda layer simply stacks outputs so both distances are available
    # to the objective
    stacked_dists = Lambda(
        lambda vects: K.stack(vects, axis=1), name="stacked_dists"
    )([positive_dist, negative_dist, tertiary_dist])

    model = Model(
        [input_anchor, input_positive, input_negative],
        stacked_dists,
        name="triple_siamese",
    )

    v_optimizer = Adam(lr=LEARNING_RATE)

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
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())
    )


def main():
    print("=> Reading file lists...")

    train_triplets = pd.read_csv(
        TRAIN_TRIPLETS,
        names=["A", "B", "C"],
        sep=" ",
        nrows=LOAD_ROWS,
        dtype=str,
    )

    test_triplets = pd.read_csv(
        TEST_TRIPLETS,
        names=["A", "B", "C"],
        sep=" ",
        nrows=LOAD_ROWS,
        dtype=str,
    )

    print("=> Creating training dataset generator")
    train_generator = image_generator(train_triplets[:TEST_SAMPLES])
    val_generator = image_generator(test_triplets[TEST_SAMPLES:])

    if False:
        print("=> Creating dataset object")

        output_types = (
            [tf.float64, tf.float64, tf.float64],
            tf.int64,
        )

        output_shapes = (
            (1),
            (T_G_NUMCHANNELS, 1),
        )

        train_data = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=output_types,
            # output_shapes=output_shapes,
        )

        val_data = tf.data.Dataset.from_generator(
            lambda: val_generator,
            output_types=output_types,
            # output_shapes=output_shapes,
        )

        print("Transform to dataset instance")
        train_data.batch(BATCH_SIZE)
        val_data.batch(BATCH_SIZE)

        print("Prepare print")
        for entry in train_data.prefetch(1).as_numpy_iterator():
            print(entry.shape)

    print("=> Create model")
    model = createNASNetLargeModel(EMBEDDING_SIZE)

    if False:
        callbacks = [
            EarlyStopping(monitor="loss", patience=3),
            ModelCheckpoint(filepath="model.{epoch:02d}-{val_loss:.2f}.h5"),
        ]

    print("Fitting model")
    model.fit(
        train_generator,
        # validation_data=val_data,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        # workers=20,
        # use_multiprocessing=True,
        # max_queue_size=10,
    )
    exit()

    ###############################################################################
    # Model inference
    ###############################################################################

    # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=False,
    #     samplewise_center=False,
    #     featurewise_std_normalization=False,
    #     samplewise_std_normalization=False,
    #     zca_whitening=False,
    #     zca_epsilon=1e-06,
    #     rotation_range=0,
    #     width_shift_range=0.0,
    #     height_shift_range=0.0,
    #     brightness_range=None,
    #     shear_range=0.0,
    #     zoom_range=0.0,
    #     channel_shift_range=0.0,
    #     fill_mode="nearest",
    #     cval=0.0,
    #     horizontal_flip=False,
    #     vertical_flip=False,
    #     rescale=None,
    #     preprocessing_function=None,
    #     data_format=None,
    #     validation_split=0.01,
    #     dtype=None,
    # )

    # train_generator = train_datagen.flow_from_directory(
    #     "data/food",
    #     target_size=(T_G_WIDTH, T_G_HEIGHT),
    #     batch_size=32,
    #     classes=np.random.randint(2, size=(1, 2, len(anchors_t))).T
    #     class_mode=None,
    # )


if __name__ == "__main__":
    main()
