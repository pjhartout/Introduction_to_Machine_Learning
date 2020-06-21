#!/usr/bin/env python
# coding: utf-8

"""The goal of this script is to make sure that we select the best pretrained
model to train it for our purposes.

We will loop through all existing constructs of neural networks proposed by
Keras and evaluate them. Following the selection process, we will then make sure
that we train this model.
"""

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


import sys
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow
from tensorflow.python.client import device_lib

T_G_WIDTH = 150
T_G_HEIGHT = 150
T_G_NUMCHANNELS = 3
CHUNKSIZE = 4096
BATCHSIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 1
MIN_EPOCHS = 10  # minimal number of epochs before early stopping takes effect.
EMBEDDING_SIZE = 400

train_triplets = pd.read_csv("data/train_triplets.txt", names=["A", "B", "C"], sep=" ")
test_triplets = pd.read_csv("data/test_triplets.txt", names=["A", "B", "C"], sep=" ")

for column in ["A", "B", "C"]:
    train_triplets[column] = train_triplets[column].astype(str)
    test_triplets[column] = test_triplets[column].astype(str)
    train_triplets[column] = train_triplets[column].apply(lambda x: x.zfill(5))
    test_triplets[column] = test_triplets[column].apply(lambda x: x.zfill(5))

train_images = train_triplets.sample(frac=1)


def createXceptionModel(emb_size):
    # Initialize an Xceptio Model pretrained on imagenet
    xception_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    xception_model = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=xception_input,
        input_shape=None,
        pooling=max,
    )

    # New Layers over Xception model
    net = xception_model.output
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(net)

    # model creation
    base_model = Model(xception_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape = (T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name="input_anchor")
    input_positive = kl.Input(shape=input_shape, name="input_pos")
    input_negative = kl.Input(shape=input_shape, name="input_neg")
    print(type(base_model))
    print(type(input_positive))
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


def createVGGModel(emb_size):
    # Initialize a VGG19 model
    vgg_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    vgg_model = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=vgg_input,
        input_shape=None,
        pooling=None,
    )

    # New Layers over VGG19
    net = vgg_model.output
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(net)

    # model creation
    base_model = Model(vgg_model.input, net, name="base_model")

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


def createInceptionResNetV2Model(emb_size):
    # Initialize a InceptionResNetV2 model
    inceptionresnetv2_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    inceptionresnetv2_model = keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inceptionresnetv2_input,
        input_shape=None,
        pooling=None,
    )

    # New Layers over InceptionResNetV2
    net = inceptionresnetv2_model.output
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(net)

    # model creation
    base_model = Model(inceptionresnetv2_model.input, net, name="base_model")

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


def createInceptionResNetV2Model(emb_size):
    # Initialize a InceptionResNetV2 model
    inceptionresnetv2_input = kl.Input(shape=(T_G_WIDTH, T_G_HEIGHT, T_G_NUMCHANNELS))
    inceptionresnetv2_model = keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inceptionresnetv2_input,
        input_shape=None,
        pooling=None,
    )

    # New Layers over InceptionResNetV2
    net = inceptionresnetv2_model.output
    net = kl.GlobalAveragePooling2D(name="gap")(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation="relu", name="t_emb_1")(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name="t_emb_1_l2norm")(net)

    # model creation
    base_model = Model(inceptionresnetv2_model.input, net, name="base_model")

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

def createNASNetLargeModel(emb_size):
    # Initialize a InceptionResNetV2 model
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


def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT, T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.vgg16.preprocess_input(
        t_image, data_format="channels_last"
    )
    return t_image


# load file names
for main_dir, subdir, file in os.walk(r"data/food/"):
    list_dir = file[:]

# load the image
img_array = {}
for file in tqdm(list_dir):
    img = t_read_image(os.path.join("data/food", file))
    img_array[file.split(".jpg")[0]] = img_to_array(img)

list_of_models = []
# This creates Xception.
#list_of_models.append(createXceptionModel(300))
#list_of_models.append(createVGGModel(300))
#list_of_models.append(createInceptionResNetV2Model(300))




print("Getting anchors train ...")
anchors_train = [img_array[img] for img in np.array(train_images["A"])]
print("Getting positives train ...")
positives_train = [img_array[img] for img in np.array(train_images["B"])]
print("Getting negatives train ...")
negatives_train = [img_array[img] for img in np.array(train_images["C"])]


total_t_ch = int(np.ceil(len(anchors_train) / float(CHUNKSIZE)))



anchors_t = anchors_train #[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
positives_t = positives_train #[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
negatives_t = negatives_train #[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
Y_train = np.random.randint(2, size=(1, 2, len(anchors_t))).T

model = createNASNetLargeModel(EMBEDDING_SIZE)

nasnet_large_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath="model.{epoch:02d}-{val_loss:.2f}.h5")
]


model.fit(
    [anchors_t, positives_t, negatives_t],
    Y_train,
    epochs=EPOCHS,
    batch_size=BATCHSIZE,
    validation_split=0.01,
    callbacks=nasnet_large_callbacks
)


# serialize model to JSON
model_json = cnn_model.to_json()
with open("model_nasnetlarge.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_model.save_weights("model_nasnetlarge.h5")
print("Saved model to disk")

# Now we want to generate the output 0, for each triplet of image on the validation to get the score
print("Getting anchors test ...")
anchors_val = [img_array[img] for img in np.array(test_triplets["A"])]
print("Getting first images ...")
first_val = [img_array[img] for img in np.array(test_triplets["B"])]
print("Getting second test ...")
second_val = [img_array[img] for img in np.array(test_triplets["C"])]

total_v_ch = int(np.ceil(len(anchors_val) / float(CHUNKSIZE)))
# for each chunk we have to compute the embedding and the distance to the closest neighbour.
predictions_list = []
errors = 0
for v in tqdm(range(0, total_v_ch)):
    anchors_val_chunk = anchors_val[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    first_val_chunk =first_val[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    second_val_chunk = second_val[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    predictions = cnn_model.predict([anchors_val_chunk, first_val_chunk, second_val_chunk], batch_size=BATCHSIZE)
    for distance in predictions:
        predictions_list.append(np.argmin(np.array([distance[1], distance[0]])))

predictions_array = np.asarray(predictions_list)

np.savetxt('predictions.txt', predictions_array, fmt='%d', delimiter='\n')




#val_accuracies = []
# for model in list_of_models:
#     for e in tqdm(range(0, EPOCHS)):
#         chunks_val_acc = []
#         for t in tqdm(range(0, total_t_ch)):
#             print("Epoch :{}, train chunk {}/{}".format(e, t + 1, total_t_ch))
#             anchors_t = anchors_train[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
#             positives_t = positives_train[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
#             negatives_t = negatives_train[t * CHUNKSIZE : (t + 1) * CHUNKSIZE]
#             Y_train = np.random.randint(2, size=(1, 2, len(anchors_t))).T
#           # This method does NOT use data augmentation
#             chunk_performance_data = model.fit(
#                 [anchors_t, positives_t, negatives_t],
#                 Y_train,
#                 epochs=1,
#                 batch_size=BATCHSIZE,
#                 validation_split=0.1,
#             )
#             chunks_val_acc.append(chunk_performance_data.history["val_accuracy"][0])
#             #val_accuracies.append(sum(chunks_val_acc) / len(chunks_val_acc))
#             tf.keras.backend.clear_session()
#             print(f"Accuracy average: {sum(chunks_val_acc) / len(chunks_val_acc)}")
#             # We stop if the difference between epochs is less than 0.05
#         # if len(val_accuracies) > MIN_EPOCHS:
#         #     difference_1 = np.abs(val_accuracies[-1] - val_accuracies[-2])
#         #     difference_2 = np.abs(val_accuracies[-1] - val_accuracies[-3])
#         #     difference_3 = np.abs(val_accuracies[-1] - val_accuracies[-4])
#         #     if difference_1 < 0.001 and difference_2 < 0.001 and difference_3 < 0.001:
#         #         break
#         print(
#             f"The accuracy is {sum(chunks_val_acc) / len(chunks_val_acc)} for model {model}"
#         )
