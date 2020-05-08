#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow-gpu


# In[1]:


import keras


# In[2]:


import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl
from keras.preprocessing.image import img_to_array
from keras import optimizers


# In[3]:


import sys
import os
import numpy as np
import pandas as pd
import cv2 
from tqdm.notebook import tqdm


# In[4]:


print(keras.__version__)
import tensorflow
print(tensorflow.__version__)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[5]:


T_G_WIDTH = 300
T_G_HEIGHT = 400
T_G_NUMCHANNELS = 3
CHUNKSIZE = 16
BATCHSIZE = 4


# In[6]:


"""from keras.preprocessing.image import ImageDataGenerator

# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(  rotation_range=90,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                )"""


# In[7]:


# generator function for data augmentation
"""def createDataGen(X1, X2, X3, Y, b):

    local_seed = 44
    genX1 = datagen.flow(X1,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX2 = datagen.flow(X2,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX3 = datagen.flow(X3,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()

        yield [X1i[0], X2i[0], X3i[0]], X1i[1]"""


# In[8]:


train_triplets = pd.read_csv("data/train_triplets.txt", names=["A", "B", "C"], sep=" ")
test_triplets = pd.read_csv("data/test_triplets.txt", names=["A", "B", "C"], sep=" ")

for column in ["A", "B", "C"]:
    train_triplets[column] = train_triplets[column].astype(str)
    test_triplets[column] = test_triplets[column].astype(str)
    train_triplets[column] = train_triplets[column].apply(lambda x: x.zfill(5))
    test_triplets[column] = test_triplets[column].apply(lambda x: x.zfill(5))
train_triplets.head()


# In[9]:


# split in test and training set, we take 0.3 of the dataframe and use it for testing and the rest for training
train_triplets = train_triplets.sample(frac=1)
n_test = 500
test_images = train_triplets[:n_test]
train_images = train_triplets[n_test:]


# In[10]:


def createModel(emb_size):

    # Initialize a ResNet50_ImageNet Model
    resnet_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    resnet_model = keras.applications.resnet50.ResNet50(weights=None, include_top = False, input_tensor=resnet_input)

    # New Layers over ResNet50
    net = resnet_model.output
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')

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
    list_dir = file[1:]


# In[14]:


# load the image
img_array = {}
for file in tqdm(list_dir):
    img = t_read_image(os.path.join("data/food", file))
    img_array[file.split(".jpg")[0]]=img_to_array(img)


# In[15]:


img_array


# In[16]:


# create model
# int is the embedding size 
model = createModel(300)


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


# In[19]:


numepochs = 1
total_t_ch = int(np.ceil(len(anchors_train) / float(CHUNKSIZE)))
total_v_ch = int(np.ceil(len(anchors_test) / float(CHUNKSIZE)))


# In[20]:


for e in range(0, numepochs):

        for t in range(0, total_t_ch):

            print ("Epoch :{}, train chunk {}/{}...".format(e,t+1,total_t_ch))

            print('Reading image lists ...')
            anchors_t = anchors_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
            positives_t =positives_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
            negatives_t = negatives_train[t*CHUNKSIZE:(t+1)*CHUNKSIZE]
            Y_train = np.random.randint(2, size=(1,2,len(anchors_t))).T

            print('Starting to fit ...')
            # This method does NOT use data augmentation
            model.fit([anchors_t, positives_t, negatives_t], Y_train, epochs=numepochs,  batch_size=BATCHSIZE)

            # This method uses data augmentation
            # model.fit_generator(generator=createDataGen(anchors_t,positives_t,negatives_t,Y_train,BATCHSIZE), 
             #                   steps_per_epoch=len(Y_train) / BATCHSIZE, epochs=1, 
            #                  shuffle=False, use_multiprocessing=True)
        

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
            w = float(anchors_v.shape[0]) / float(CHUNKSIZE)
            total_w = total_w + w

            curval = model.evaluate([anchors_v, positives_v, negatives_v], Y_val, batch_size=BATCHSIZE)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]

        val_res = [x / total_w for x in val_res]

        print('Validation Results: ', str(val_res))


# In[ ]:





# In[ ]:




