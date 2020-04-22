# Standard
import argparse
import numpy as np
import pandas as pd
import os
import glob
import pickle
import time

# Keras import
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# This project
from common import make_directory, set_batches_per_epoch
from network import init_model, init_adam
from train import train

if __name__ == "__main__":

    # Setup directory tree
    make_directory('baseline/')
    make_directory('baseline/model/')
    make_directory('baseline/metrics/')
    make_directory('baseline/params/')

    # Determine trail index
    trial_id = 0 
    models = glob.glob('baseline/model/*.h5')
    for model in models:
        index = int(model.split('.')[1])
        if index >= trial_id:
            trial_id = index + 1
    print('Performing trial {}'.format(trial_id))
    savename = 'baseline/model/model.{}.h5'.format(trial_id)
    
    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
    X_train, X_test = X_train / 255, X_test / 255
    
    # Baseline setup
    params = {
        'input_shape':X_train.shape[1:],
        'output_shape':Y_train.shape[1],
        'depth':3,
        'dense_neurons':128,
        'init_filters':16,
        'use_batchnorm':True,
        'dropout':0.4,
        'batch_size':128,
        'max_epochs':100,
        'learning_rate':0.0025,
        'beta1':0.75,
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
                           patience=10, savename=savename)

    # Save if it is not saved
    if not os.path.exists(savename):
        model.save(savename)

    # Get best model 
    model = load_model(savename)
        
    data = {'train_loss':t_loss, 'valid_loss':v_loss}
    df = pd.DataFrame(data)
    df.to_csv('baseline/metrics/metrics.{}.csv'.format(trial_id), index=False)

    # Print final testing loss
    test_loss = model.evaluate(X_test[6000:], Y_test[6000:])
    print("Final testing loss {0:8.4f}".format(test_loss))

    # Top-k Accuracy
    Y_pred = np.argmax(model.predict(X_test[6000:]), axis=1)
    Y_true = np.argmax(Y_test[6000:], axis=1)
    acc = accuracy_score(Y_true, Y_pred)
    
    # Save params
    with open('baseline/params/params.{}.pkl'.format(trial_id), 'wb') as output_file:
        pickle.dump(params, output_file)

    # Write metafile
    meta_cols = ['dropout', 'batch_size', 'max_epochs', 'depth', 'dense_neurons', 'init_filters',
                 'use_batchnorm', 'learning_rate', 'beta1', 'beta2']
    meta_data = {k:v for k,v in params.items() if k in meta_cols}
    if not os.path.exists('baseline/meta.csv'):
        meta_df = pd.DataFrame({'id':trial_id, 'time':time.time(), 'test_loss':test_loss,
                                'epochs':len(t_loss), 'accuracy':acc, **meta_data}, index=[trial_id,])
        meta_df.to_csv('baseline/meta.csv', index=False)
    else:
        meta_df = pd.read_csv('baseline/meta.csv')
        new_meta = pd.DataFrame({'id':trial_id, 'time':time.time(), 'test_loss':test_loss,
                                'epochs':len(t_loss), 'accuracy':acc, **meta_data}, index=[trial_id,])
        out_df = pd.concat([meta_df, new_meta])
        out_df.to_csv('baseline/meta.csv', index=False)
