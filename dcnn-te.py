#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:17:57 2018

@author: Alexandre Romanelli, Antonio Luiz da Silva Loca
"""

import os
import sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

def centralize(D):
    mean = D.mean(axis=0)
    D_centr = D - mean
    return D_centr

def normalize(D):
    std = D.std(axis=0)
    std[std == 0] = 1
    D_centr = centralize(D)
    D_norm = D_centr / std
    return D_norm

def csvread(filename, delimiter = '\t'):
    f = open(filename, 'rb')
    reader = csv.reader(f, delimiter=delimiter)
    ncol = len(next(reader)) # Read first line and count columns
    nfeat = ncol-1
    f.seek(0)              # go back to beginning of file
    #print('ncol=', ncol)
    
    x = np.zeros(nfeat)
    X = np.empty((0, nfeat))
    y = []
    for row in reader:
        #print(row)
        for j in range(nfeat):
            x[j] = float(row[j])
            #print('j=', j, ':', x[j])
        X = np.append(X, [x], axis=0)
        label = row[nfeat]
        y.append(label)
    
    return X, y

def encodeLabel(y):
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    classname = lb.classes_
    print('lb.classes_=', lb.classes_)
    le = LabelEncoder()
    ynum = le.fit_transform(y)
    return Y, ynum, classname

def shuffleArrays(D, C):
    s = np.arange(D.shape[0])
    np.random.shuffle(s)
    d = D[s]
    c = C[s]
    return d, c
    
def dividirTreinamentoTeste(D, C):
    dados_treinamento = np.empty([0, 20, 50, 1])
    dados_teste = np.empty([0, 20, 50, 1])
    classes_treinamento = np.empty([0, 21])
    classes_teste = np.empty([0, 21])
    # dividir as 500h iniciais entre treinamento e teste (80%, 20%)
    normal = D[0:500, :, :]
    classe_normal = C[0:500, :]
    d_trein, d_teste, c_trein, c_teste = train_test_split(normal, classe_normal, train_size=0.8)
    dados_treinamento = np.append(dados_treinamento, d_trein, axis=0)
    dados_teste = np.append(dados_teste, d_teste, axis=0)
    classes_treinamento = np.append(classes_treinamento, c_trein, axis=0)
    classes_teste = np.append(classes_teste, c_teste, axis=0)
    # dividir os registros com falhas
    indInicioSimulacaoFalhas = 500
    tamEstadoNormal = 10
    tamEstadoFalha = 40
    for f in range(1, 21): # f \in [1, 20]
        # primeira simulação da falha
        indInicioFalha = tamEstadoNormal + indInicioSimulacaoFalhas + (f - 1) * (tamEstadoNormal + tamEstadoFalha)
        indFimFalha = indInicioFalha + tamEstadoFalha
        falha = D[indInicioFalha:indFimFalha,:,:,:]
        classe_falha = C[indInicioFalha:indFimFalha,:]
        # 80% para treinamento, 20% para testes
        d_trein, d_teste, c_trein, c_teste = train_test_split(falha, classe_falha, train_size=0.8)
        dados_treinamento = np.append(dados_treinamento, d_trein, axis=0)
        dados_teste = np.append(dados_teste, d_teste, axis=0)
        classes_treinamento = np.append(classes_treinamento, c_trein, axis=0)
        classes_teste = np.append(classes_teste, c_teste, axis=0)
        # percorrer as outras simulações desta falha
        for i in range(1, 10):
            indInicioFalha = indInicioFalha + i * (tamEstadoFalha + tamEstadoNormal)
            indFimFalha = indInicioFalha + tamEstadoFalha
            falha = D[indInicioFalha:indFimFalha,:,:,:]
            classe_falha = C[indInicioFalha:indFimFalha,:]
            # 80% para treinamento, 20% para testes
            d_trein, d_teste, c_trein, c_teste = train_test_split(falha, classe_falha, train_size=0.8)
            dados_treinamento = np.append(dados_treinamento, d_trein, axis=0)
            dados_teste = np.append(dados_teste, d_teste, axis=0)
            classes_treinamento = np.append(classes_treinamento, c_trein, axis=0)
            classes_teste = np.append(classes_teste, c_teste, axis=0)

    dados_treinamento, classes_treinamento = shuffleArrays(dados_treinamento, classes_treinamento)
    dados_teste, classes_teste = shuffleArrays(dados_teste, classes_teste)
    return dados_treinamento, classes_treinamento, dados_teste, classes_teste

# Data
nome_arq = "TE-simulations/out/te_simulation_{:02d}_50.csv"
X = np.empty([0, 53])
y = []
sys.stdout.write("Lendo arquivos de dados: {:3d}%".format(0))
for i in range(1, 51):
    X_, y_ = csvread(nome_arq.format(i))
    X = np.append(X, X_, axis=0)
    y = sum([y, y_], [])
    sys.stdout.write("\rLendo arquivos de dados: {:3d}%".format(i * 2))
    
print(" - Leitura concluída.\n")

Y, ynum, classname = encodeLabel(y)

std = X.std(axis=0)
print(std)

XMV = ['D feed flow (stream 2)',
       'E feed flow (stream 3)',
       'A feed flow (stream 1)',
       'A and C feed flow (stream 4)',
       'Compressor recycle valve',                  # ---------------------------> removido no paper
       'Purge valve (stream 9)',
       'Separator pot liquid flow (stream 10)',
       'Stripper liquid product flow (stream 11)',
       'Stripper steam valve',                      # ---------------------------> removido no paper
       'Reactor cooling water flow',
       'Condenser cooling water flow',
       'Agitator speed']                            # ---------------------------> removido no paper
XMEAS = ['Input Feed - A feed (stream 1)',
         'Input Feed - D feed (stream 2)',
         'Input Feed - E feed (stream 3)',
         'Input Feed - A and C feed (stream 4)',
         'Reactor feed rate (stream 6)',
         'Reactor pressure',
         'Reactor level',
         'Reactor temperature',
         'Separator - Product separator temperature',
         'Separator - Product separator level',
         'Separator - Product separator pressure',
         'Separator - Product separator underflow (stream 10)',
         'Stripper level',
         'Stripper pressure',
         'Stripper underflow (stream 11)',
         'Stripper temperature',
         'Stripper steam flow',
         'Miscellaneous - Recycle flow (stream 8)',
         'Miscellaneous - Purge rate (stream 9)',
         'Miscellaneous - Compressor work',
         'Miscellaneous - Reactor cooling water outlet temperature',
         'Miscellaneous - Separator cooling water outlet temperature',
         'Reactor Feed Analysis - Component A',
         'Reactor Feed Analysis - Component B',
         'Reactor Feed Analysis - Component C',
         'Reactor Feed Analysis - Component D',
         'Reactor Feed Analysis - Component E',
         'Reactor Feed Analysis - Component F',
         'Purge gas analysis - Component A',
         'Purge gas analysis - Component B',
         'Purge gas analysis - Component C',
         'Purge gas analysis - Component D',
         'Purge gas analysis - Component E',
         'Purge gas analysis - Component F',
         'Purge gas analysis - Component G',
         'Purge gas analysis - Component H',
         'Product analysis -  Component D',
         'Product analysis - Component E',
         'Product analysis - Component F',
         'Product analysis - Component G',
         'Product analysis - Component H']

featname = XMV + XMEAS

features = np.array(range(0, 53))

labels = ynum
classes = classname
classlabels = np.unique(ynum)

# Remover colunas descartadas no paper
X = np.delete(X, [4, 8, 11], axis=1)
print(X.shape)

# centralizar e normalizar os dados
X = normalize(X)
std = X.std(axis=0)
print(std)

# reajustar os dados para ficarem organizados em grupos de 1h
X = np.reshape(X, (-1, 20, 50, 1))
# manter em Y apenas os primeiros valores de cada grupo de 20 linhas (= 1h)
Y = np.reshape(Y, (-1, 20, 21))[:,0,:]

print 'X.shape: ', X.shape
print 'Y.shape: ', Y.shape

# Seguindo o artigo, separar entre train data (80%) e test data (80%)
X_train, Y_train, X_test, Y_test = dividirTreinamentoTeste(X, Y)

print 'X_train.shape: ', X_train.shape
print 'Y_train.shape: ', Y_train.shape
print 'X_test.shape: ', X_test.shape
print 'Y_test.shape: ', Y_test.shape

# -----------------------------------------------------------------------------
# Deep Convolutional Neural Network
# -----------------------------------------------------------------------------
batch_size = 128
epochs = 50
# TE simulation: 21 classes
num_classes = 21

# Model 7
cnn_model = Sequential()
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

cnn_model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
cnn_model.summary()

cnn_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
