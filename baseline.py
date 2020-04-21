# Standard
import argparse
import numpy as np
import pandas as pd
import os

# Keras import
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

# This project
from network import init_model
from train import train

if __name__ == "__main__":

    # Baseline setup
    params = {
        'input_shape':(32,32,3),
        'output_shape':10,
        'depth':3,
        'init_filters':32,
        'use_batchnorm':False
    }
    model = init_model(params)
    print(model.summary())

    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

    
