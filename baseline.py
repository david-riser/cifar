# Standard
import argparse
import numpy as np
import pandas as pd
import os

# Keras import
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

# This project
from common import make_directory, set_batches_per_epoch
from network import init_model, init_adam
from train import train

if __name__ == "__main__":


    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

    # Baseline setup
    params = {
        'input_shape':X_train.shape[1:],
        'output_shape':Y_train.shape[1],
        'depth':3,
        'dense_neurons':128,
        'init_filters':32,
        'use_batchnorm':True,
        'dropout':0.2,
        'batch_size':32,
        'max_epochs':3,
        'learning_rate':0.001,
        'beta1':0.9,
        'beta2':0.999
    }
    params['batches_per_epoch'] = set_batches_per_epoch(params['batch_size'])
    print(params)

    model = init_model(params)
    adam = init_adam(params)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    print(model.summary())

    # Train but reserve the last little bit of data for
    # a final testing set.
    t_loss, v_loss = train(model, params, X_train, Y_train, X_test[:6000], Y_test[:6000],
                           patience=10, savename='baseline/model/model.h5')

    make_directory('baseline/')
    make_directory('baseline/model/')
    make_directory('baseline/metrics/')

    if not os.path.exists('baseline/model/model.h5'):
        model.save('baseline/model/model.h5')

    data = {'train_loss':t_loss, 'valid_loss':v_loss}
    df = pd.DataFrame(data)
    df.to_csv('baseline/metrics/metrics.csv', index=False)

    # Print final testing loss
    test_loss = model.evaluate(X_test[6000:], Y_test[6000:])
    print("Final testing loss {0:8.4f}".format(test_loss))
