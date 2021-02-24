# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""

import keras.utils
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import Conv2D, Input, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model

def square(x): 
    return K.square(x) 

def log(x): 
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 

def createModel(nb_classes = 4, Chans = 22, Samples = 1000, dropoutRate = 0.5, init='he_uniform'): 
    my_shape = (Samples, Chans, 1)
    if K.image_data_format() == 'channels_first':
        my_shape = (1, Samples, Chans)
    input_main   = Input(my_shape) 
    block = Conv2D(40, (45, 1), strides = (2, 1), use_bias = False,
                    input_shape = my_shape, 
                    kernel_constraint = max_norm(3.0, axis=(0,1,2)))(input_main) 
    block = Conv2D(40, (1, Chans), use_bias=False,  name='channelConv')(block)
    block = BatchNormalization(epsilon=1e-05, momentum=0.1)(block) 
    block = keras.layers.Activation(square)(block)
    block  = keras.layers.AveragePooling2D(pool_size=(45, 1), strides=(8, 1))(block) 
    block  = keras.layers.Activation(log)(block) 
    block  = Dropout(dropoutRate)(block)
    flatten = Flatten()(block) 
    dense   = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten) 
    softmax = keras.layers.Activation('softmax')(dense)      
    return Model(inputs=input_main, outputs=softmax) 