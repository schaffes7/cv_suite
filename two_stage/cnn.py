import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


def cnn_regressor(img_shape, opt = 'adagrad', loss = 'mean_squared_error', final_act = 'sigmoid', lr = 0.001, dr = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = img_shape, activation = 'relu', data_format = 'channels_last'))
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu'))              # DENSE
    model.add(Dropout(dr))                                # DROP
    model.add(Dense(64, activation = 'relu'))               # DENSE
    model.add(Dropout(dr))                                # DROP
    model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt == 'rms':
        optimizer = keras.optimizers.RMSprop(lr = lr, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt == 'sgd':
        optimizer = keras.optimizers.SGD(lr = lr, decay = 1e-6)
    if opt == 'adam':
        optimizer = keras.optimizers.Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt == 'adagrad':
        optimizer = keras.optimizers.Adagrad(lr = lr, epsilon = None, decay = 0.0)
    if opt == 'adadelta':
        optimizer = keras.optimizers.Adadelta(lr = lr, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    return model


def cnn_classifier(img_shape, n_classes = 3, ch = 3, opt = 'adagrad', loss = 'categorical_crossentropy', final_act = 'softmax', lr = 0.001, dr = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = img_shape, activation = 'relu', data_format = 'channels_last'))
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(Conv2D(32, (3,3), activation = 'relu'))      # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu'))              # DENSE
    model.add(Dropout(dr))                                # DROP
    model.add(Dense(64, activation = 'relu'))               # DENSE
    model.add(Dropout(dr))                                # DROP
    model.add(Dense(n_classes, activation = final_act))             # DENSE
    # Define Optimizer
    if opt == 'rms':
        optimizer = keras.optimizers.RMSprop(lr = lr, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt == 'sgd':
        optimizer = keras.optimizers.SGD(lr = lr, decay = 1e-6)
    if opt == 'adam':
        optimizer = keras.optimizers.Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt == 'adagrad':
        optimizer = keras.optimizers.Adagrad(lr = lr, epsilon = None, decay = 0.0)
    if opt == 'adadelta':
        optimizer = keras.optimizers.Adadelta(lr = lr, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    return model