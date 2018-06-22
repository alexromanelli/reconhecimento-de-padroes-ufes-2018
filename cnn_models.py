#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:59:08 2018

@author: alex
"""

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def prepararModeloCNN(indiceModelo, num_classes):
    cnn_model = Sequential()
    if indiceModelo == 1:
        # preparar modelo 1
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 2:
        #preparar modelo 2
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 3:
        #preparar modelo 3
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 4:
        #preparar modelo 4
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 5:
        #preparar modelo 5
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 6:
        #preparar modelo 6
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 7:
        #preparar modelo 7
        cnn_model.add(Conv2D(64, 
                             kernel_size=(3, 3),
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 1),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 8:
        #preparar modelo 8
        cnn_model.add(Conv2D(64, 
                             kernel_size=(3, 3),
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(1, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 9:
        #preparar modelo 9
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 10:
        #preparar modelo 10
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    elif indiceModelo == 11:
        #preparar modelo 11
        cnn_model.add(Conv2D(64, 
                             kernel_size=(3, 3),
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 1),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))
    else:
        #preparar modelo 12
        cnn_model.add(Conv2D(64, 
                             kernel_size=(3, 3),
                             activation='linear',
                             input_shape=(20, 50, 1),
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Conv2D(64, (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(Conv2D(128, 
                             (3, 3), 
                             activation='linear',
                             padding='same'))
        cnn_model.add(LeakyReLU(alpha=0.1))                  
        cnn_model.add(MaxPooling2D(pool_size=(2, 1),
                                   strides=2,
                                   padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(300, 
                            activation='linear'))
        cnn_model.add(LeakyReLU(alpha=0.1))           
        cnn_model.add(Dropout(0.5))       
        cnn_model.add(Dense(num_classes, 
                            activation='softmax'))

    cnn_model.compile(loss=keras.losses.categorical_crossentropy, 
                      #loss=keras.losses.mean_squared_error, 
                      optimizer=keras.optimizers.Adam(),
                      # https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/
                      #optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
    return cnn_model

