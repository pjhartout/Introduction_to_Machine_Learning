#!/usr/bin/env python
# coding: utf-8

# Implementation of triplet loss using keras inspired by https://github.com/noelcodella/tripletloss-keras-tensorflow

# In[2]:


import keras
import tensorflow as tf
import tensorflow_addons as tfa


# In[3]:


import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl
from keras.preprocessing.image import img_to_array
from keras import optimizers


# In[4]:


import sys
import os
import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm


# In[5]:


print(keras.__version__)
import tensorflow
print(tensorflow.__version__)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[6]:


T_G_WIDTH = 150
T_G_HEIGHT = 150
T_G_NUMCHANNELS = 3
CHUNKSIZE = 32
BATCHSIZE = 2048


# In[7]:


train_triplets = pd.read_csv("data/train_triplets.txt", names=["A", "B", "C"], sep=" ")
test_triplets = pd.read_csv("data/test_triplets.txt", names=["A", "B", "C"], sep=" ")

for column in ["A", "B", "C"]:
    train_triplets[column] = train_triplets[column].astype(str)
    test_triplets[column] = test_triplets[column].astype(str)
    train_triplets[column] = train_triplets[column].apply(lambda x: x.zfill(5))
    test_triplets[column] = test_triplets[column].apply(lambda x: x.zfill(5))
train_triplets.head()


# In[8]:


# split in test and training set, we take 0.3 of the dataframe and use it for testing and the rest for training
train_triplets = train_triplets.sample(frac=1)
n_test = 500
test_images = train_triplets[:n_test]
train_images = train_triplets[n_test:]


# In[9]:


# def createCNNModel(emb_size):

#     # Initialize a ResNet50_ImageNet Model
# #     resnet_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(filters=10, kernel_size=2, padding='same', activation='relu', input_shape=(T_G_HEIGHT,T_G_WIDTH,3)),
#         tf.keras.layers.MaxPooling2D(pool_size=2),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(emb_size, activation=None), # No activation on final dense layer
#         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
#     ])

#     print(type(model))
#     # triplet framework, shared weights
#     input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
#     input_anchor = kl.Input(shape=input_shape, name='input_anchor')
#     input_positive = kl.Input(shape=input_shape, name='input_pos')
#     input_negative = kl.Input(shape=input_shape, name='input_neg')
#     print(type(model))
#     print(type(input_positive))
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tfa.losses.TripletSemiHardLoss())
#     net_anchor = model(input_anchor)
#     net_positive = model(input_positive)
#     net_negative = model(input_negative)

#     # The Lamda layer produces output using given function. Here its Euclidean distance.
#     positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
#     negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
#     tertiary_dist = kl.Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

#     # This lambda layer simply stacks outputs so both distances are available to the objective
#     stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

#     model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

#     v_optimizer = optimizers.Adam(lr=0.001)

#     model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

#     return model
# createCNNModel(100)


# In[10]:


def createResNetModel(emb_size):

    # Initialize a ResNet50_ImageNet Model
    xception_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    xception_model = keras.applications.Xception(include_top=False,
        weights=None,
        input_tensor=xception_input,
        input_shape=None,
        pooling=max,
    )

    # New Layers over ResNet50
    net = xception_model.output
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)
    
    # model creation
    base_model = Model(xception_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')
    print(type(base_model))
    print(type(input_positive))
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = kl.Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    v_optimizer = optimizers.Adam(lr=0.001)

    model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

    return model


# In[11]:


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# In[12]:


def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')

    return t_image


# In[13]:


# load file names
for direc, subdir, file in os.walk(r"data/food"):
    list_dir = file[:]


# In[14]:


# load the image
img_array = {}
for file in tqdm(list_dir):
#     print(file)
    img = t_read_image(os.path.join("data/food", file))
    img_array[file.split(".jpg")[0]]=img_to_array(img)


# In[15]:


# create model
# int is the embedding size 
# resnet_model = createResNetModel(300)


# In[16]:


use_pretrained_model = False
if use_pretrained_model == False:
    cnn_model = createResNetModel(300) 
else:
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    cnn_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    cnn_model.load_weights("Xception.h5")
    cnn_model.compile(optimizer=optimizers.Adam(lr=0.001), loss=triplet_loss, metrics=[accuracy])
    print("Loaded model from disk")


# In[ ]:





# In[17]:


print("Getting anchors train ...")
anchors_train = [img_array[img] for img in np.array(train_images["A"])]
print("Getting positives train ...")
positives_train = [img_array[img] for img in np.array(train_images["B"])]
print("Getting negatives train ...")
negatives_train = [img_array[img] for img in np.array(train_images["C"])]


# In[18]:


print("Getting anchors test ...")
anchors_test = [img_array[img] for img in np.array(test_images["A"])]
print("Getting positives test ...")
positives_test = [img_array[img] for img in np.array(test_images["B"])]
print("Getting negatives test ...")
negatives_test = [img_array[img] for img in np.array(test_images["C"])]


# In[22]:


numepochs = 100
total_t_ch = int(np.ceil(len(anchors_train) / float(CHUNKSIZE)))
total_v_ch = int(np.ceil(len(anchors_test) / float(CHUNKSIZE)))


# In[ ]:


for e in tqdm(range(0, numepochs)):
    for t in tqdm(range(0, total_t_ch)):
        print ("Epoch :{}, train chunk {}/{}".format(e,t+1,total_t_ch))
        anchors_t = anchors_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
        positives_t =positives_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
        negatives_t = negatives_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
        Y_train = np.random.randint(2, size=(1,2,len(anchors_t))).T
        # This method does NOT use data augmentation
        cnn_model.fit([anchors_t, positives_t, negatives_t], Y_train, epochs=1,  batch_size=BATCHSIZE)        


# In[ ]:


# In case the validation images don't fit in memory, we load chunks from disk again. 
val_res = [0.0, 0.0]
total_w = 0.0
for v in range(0, total_v_ch):

    print('Loading validation image lists ...')
    print ("Epoch :{}, train chunk {}/{}...".format(e,v+1,total_v_ch))
    anchors_v = anchors_test[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    positives_v =positives_test[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    negatives_v = negatives_test[v*CHUNKSIZE:(v+1)*CHUNKSIZE]
    Y_val = np.random.randint(2, size=(1,2,len(anchors_v))).T

    # Weight of current validation measurement. 
    # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
    w = float(len(anchors_v)) / float(CHUNKSIZE)
    total_w = total_w + w

    curval = cnn_model.evaluate([anchors_v, positives_v, negatives_v], Y_val, batch_size=BATCHSIZE)
    val_res[0] = val_res[0] + w*curval[0]
    val_res[1] = val_res[1] + w*curval[1]
#     print(pd.Series(np.argmax(val_res)).value_counts())
val_res = [x / total_w for x in val_res]

print('Validation Results: ', str(val_res))


# In[ ]:


# serialize model to JSON
model_json = cnn_model.to_json()
with open("model_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_model.save_weights("Xception_2.h5")
print("Saved model to disk")
 


# In[ ]:





# In[ ]:


# Now we want to generate the output 0, for each triplet of image on the validation to get the score


# In[ ]:


print("Getting anchors test ...")
anchors_val = [img_array[img] for img in np.array(test_triplets["A"])]
print("Getting first images ...")
first_val = [img_array[img] for img in np.array(test_triplets["B"])]
print("Getting second test ...")
second_val = [img_array[img] for img in np.array(test_triplets["C"])]


# In[ ]:


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


# In[ ]:


predictions_array = np.asarray(predictions_list)


# In[ ]:


np.savetxt('predictions.txt', predictions_array, fmt='%d', delimiter='\n')


# In[ ]:





# In[ ]:




