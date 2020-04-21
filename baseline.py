# Standard
import argparse
import numpy as np
import pandas as pd
import os

# Keras import
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

# This project
from network import init_model, init_adam
from train import train

def set_batches_per_epoch(batch_size):
    """ Only powers of two please. """
    total = 16 * 1024
    return int(total / batch_size)
    
if __name__ == "__main__":


    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

    # Baseline setup
    params = {
        'input_shape':X_train.shape[1:],
        'output_shape':Y_train.shape[1],
        'depth':3,
        'init_filters':32,
        'use_batchnorm':True,
        'batch_size':32,
        'max_epochs':1,
        'learning_rate':0.001,
        'beta1':0.99,
        'beta2':0.99
    }
    params['batches_per_epoch'] = set_batches_per_epoch(params['batch_size'])
    print(params)

    model = init_model(params)
    adam = init_adam(params)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    print(model.summary())

    # Train but reserve the last little bit of data for
    # a final testing set.
    t_loss, v_loss = train(model, params, X_train, Y_train, X_test[:6000], Y_test[:6000])
