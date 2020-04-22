# Standard
import argparse
import numpy as np
import pandas as pd
import os
import glob
import pickle
import sherpa
import time

from copy import deepcopy

# Keras import
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# This project
from common import make_directory, set_batches_per_epoch
from network import init_model, init_adam
from parameters import build_sherpa_parameter_space
from train import train

if __name__ == "__main__":

    algorithm = 'gpyopt'
    
    # Setup directory tree
    make_directory('{}/'.format(algorithm))
    make_directory('{}/model/'.format(algorithm))
    make_directory('{}/metrics/'.format(algorithm))
    make_directory('{}/params/'.format(algorithm))
    
    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
    X_train, X_test = X_train / 255, X_test / 255
    
    # Baseline setup
    params = build_sherpa_parameter_space()
    extra_params = {
        'input_shape':X_train.shape[1:],
        'output_shape':Y_train.shape[1],
        'max_epochs':100,
    }

    sherpa_algo = sherpa.algorithms.GPyOpt(max_num_trials=50)
    study = sherpa.Study(parameters=params, algorithm=sherpa_algo, lower_is_better=True)
    for trial in study:
        trial_id = trial.id
        savename = '{}/model/model.{}.h5'.format(algorithm, trial_id)
        pars = deepcopy(trial.parameters) 
        pars['batches_per_epoch'] = set_batches_per_epoch(pars['batch_size'])

        for k, v in extra_params.items():
            pars[k] = v
        
        model = init_model(pars)
        adam = init_adam(pars)
        model.compile(optimizer=adam, loss='categorical_crossentropy')

        # Train but reserve the last little bit of data for
        # a final testing set.
        t_loss, v_loss = train(model, pars, X_train, Y_train, X_test[:6000], Y_test[:6000],
                               patience=10, savename=savename)

        # Save if it is not saved
        if not os.path.exists(savename):
            model.save(savename)

        # Get best model 
        model = load_model(savename)
        
        data = {'train_loss':t_loss, 'valid_loss':v_loss}
        df = pd.DataFrame(data)
        df.to_csv('{}/metrics/metrics.{}.csv'.format(algorithm, trial_id), index=False)

        # Print final testing loss
        test_loss = model.evaluate(X_test[6000:], Y_test[6000:])
        print("Final testing loss {0:8.4f}".format(test_loss))

        # Top-k Accuracy
        Y_pred = np.argmax(model.predict(X_test[6000:]), axis=1)
        Y_true = np.argmax(Y_test[6000:], axis=1)
        acc = accuracy_score(Y_true, Y_pred)
    
        # Save params
        with open('{}/params/params.{}.pkl'.format(algorithm, trial_id), 'wb') as output_file:
            pickle.dump(pars, output_file)

        # Write metafile
        meta_cols = ['dropout', 'batch_size', 'max_epochs', 'depth', 'dense_neurons', 'init_filters',
                     'use_batchnorm', 'learning_rate', 'beta1', 'beta2']
        meta_data = {k:v for k,v in pars.items() if k in meta_cols}
        if not os.path.exists('{}/meta.csv'.format(algorithm)):
            meta_df = pd.DataFrame({'id':trial_id, 'time':time.time(), 'test_loss':test_loss,
                                    'epochs':len(t_loss), 'accuracy':acc, **meta_data}, index=[trial_id,])
            meta_df.to_csv('{}/meta.csv'.format(algorithm), index=False)
        else:
            meta_df = pd.read_csv('{}/meta.csv'.format(algorithm))
            new_meta = pd.DataFrame({'id':trial_id, 'time':time.time(), 'test_loss':test_loss,
                                     'epochs':len(t_loss), 'accuracy':acc, **meta_data}, index=[trial_id,])
            out_df = pd.concat([meta_df, new_meta])
            out_df.to_csv('{}/meta.csv'.format(algorithm), index=False)
