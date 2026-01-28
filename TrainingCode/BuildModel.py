# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import regularizers


def LSTM(inShape1, inShape2, lays=3, nodes=50, optimizer='Adam', lr=0.001, Drop=False,
         dPer=0.5, regular=False, rCoef=0.001, activate='tanh'):

    # Print setup
    print("Building the Network")
    print("Number of layers: ", lays)
    print("Neurons per layer: ", nodes)
    if (Drop):
        print("Building model with dropout")
        print("Dropout percentage", dPer)
    if (regular):
        print("Building model with L2 regularization")
        print("Regularization coefficient", rCoef)

    # Create model
    model = models.Sequential()

    # Dont return the sequence if only 1 LSTM layer
    if (lays == 1):
        # Dont return sequence
        returnSeq = False
    else:
        returnSeq = True
    # Add model layers
    if (regular):
        print('Adding regularization')
        model.add(layers.LSTM(nodes, kernel_regularizer=regularizers.l2(rCoef), activation=activate,
                              return_sequences=returnSeq, input_shape=(inShape1, inShape2)))
    elif (Drop):
        model.add(layers.LSTM(nodes, activation=activate, dropout=dPer,
                              return_sequences=returnSeq, input_shape=(inShape1, inShape2)))
    else:
        model.add(layers.LSTM(nodes, activation=activate,
                              return_sequences=returnSeq, input_shape=(inShape1, inShape2)))
    if (lays > 1):
        for i in range(lays - 1):
            print('Adding Hidden layer: ', i + 1)
            if (i == lays - 2):
                # Dont return sequence
                returnSeq = False
            else:
                returnSeq = True
            if (regular):
                print('Adding regularization')
                model.add(layers.LSTM(nodes, kernel_regularizer=regularizers.l2(
                    rCoef), return_sequences=returnSeq, activation=activate))
            elif (Drop):
                model.add(layers.LSTM(nodes, dropout=dPer,
                                      return_sequences=returnSeq, activation=activate))
            else:
                model.add(layers.LSTM(nodes, return_sequences=returnSeq, activation=activate))
    else:
        i = 0
    print('Adding Output layer: ', i + 2)
    model.add(layers.Dense(1))

    # Define the optimizer
    if (optimizer == 'SGD'):
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif (optimizer == 'RMSprop'):
        opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
    elif (optimizer == 'Adagrad'):
        opt = optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif (optimizer == 'Adadelta'):
        opt = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    elif (optimizer == 'Adam'):
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=0.0, amsgrad=False)
    elif (optimizer == 'Adamax'):
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif (optimizer == 'Nadam'):
        opt = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999,
                               epsilon=None, schedule_decay=0.004)
    else:
        print('Bad optimizer option...')
        exit()

    # Compile the model
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

    return model
