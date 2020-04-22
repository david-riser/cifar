import argparse
import numpy as np
import os
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(model, params, X_train, Y_train, X_test, Y_test,
          patience=10, savename='model.h5'):
    """ Train the model according to the parameters given, this
    function is meant to work with sherpa. 

    params (dict):
    -------
    This must contain the following variables.
    - max_epochs
    - batches_per_epoch
    - batch_size
    """

    # Patience and Model Checkpointing
    down_rounds = 0
    best_loss = np.inf
    
    t_loss, v_loss = [], []
    for epoch in range(params['max_epochs']):
        index = np.random.choice(np.arange(X_train.shape[0]), size=params['batch_size']*params['batches_per_epoch'])
        batch_loss = 0.
        for batch in range(params['batches_per_epoch']):
            batch_index = index[batch * params['batch_size']:(batch+1) * params['batch_size']]
            batch_loss += model.train_on_batch(X_train[batch_index], Y_train[batch_index])

        # Training loss from above
        t_loss.append(batch_loss / params['batches_per_epoch'])

        # Evaluate validation set
        v_loss.append(model.evaluate(X_test, Y_test))

        # Model checkpoint and early stopping.
        if v_loss[epoch] < best_loss:
            best_loss = v_loss[epoch]
            down_rounds = 0
            model.save(savename)
            print("[New Best] Epoch {0}, Training Loss: {1:8.4f}, Testing Loss: {2:8.4f}".format(
                epoch, t_loss[epoch], v_loss[epoch]))
        else:
            down_rounds += 1

        if down_rounds >= patience:
            print("Earlying stopping at epoch {}.".format(epoch))
            break

        # This line is important for doing searches because
        # we keep wasting time in spaces where nothing is
        # happening at all.
        if (epoch == 3) and (down_rounds == 2):
            print("Earlying stopping because nothing is happening.")
            break
        
    return t_loss, v_loss

def train_model_with_augmentations(model, datagen, X_test, Y_test,
                                   epochs=100, batch_size=256,
                                   batches_per_epoch=4, patience=15, savename="model.h5"):
    """ Train the model until it can not be trained anymore! """

    train_loss = []
    valid_loss = []
    train_top1_acc = []
    valid_top1_acc = []
    train_top5_acc = []
    valid_top5_acc = []
    down_rounds = 0
    best_loss = np.inf
    Y_test_cat = to_categorical(Y_test)

    # Create a data flow
    data_flow = datagen.flow(X_train, Y_train,
                             batch_size=batch_size,
                             shuffle=True)

    for epoch in range(epochs):
        
        batch_loss = []
        batch_top1 = []
        batch_top5 = []
        for batch in range(batches_per_epoch):
            X_batch, Y_batch = next(data_flow)
            step_loss = model.train_on_batch(X_batch, Y_batch)
            batch_loss.append(step_loss)
            batch_top1.append(
                np.mean(top_k_categorical_accuracy(Y_batch, model.predict(X_batch), k=1)))
            batch_top5.append(
                np.mean(top_k_categorical_accuracy(Y_batch, model.predict(X_batch), k=5)))

        # End of epoch logging
        Y_pred = model.predict(X_test)
        valid_top1_acc.append(np.mean(top_k_categorical_accuracy(Y_test_cat, model.predict(X_test), k=1)))
        valid_top5_acc.append(np.mean(top_k_categorical_accuracy(Y_test_cat, model.predict(X_test), k=5)))
        train_top1_acc.append(np.mean(batch_top1))
        train_top5_acc.append(np.mean(batch_top5))
        train_loss.append(np.mean(batch_loss))
        valid_loss.append(model.evaluate(X_test, Y_test_cat))

        # Model checkpoint and early stopping.
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            down_rounds = 0
            model.save(savename)
            print(
                "[New Best] Epoch {0}, Training Loss: {1:8.4f}, Testing Loss: {2:8.4f}, Training Top-1: {3:8.4f}, Testing Top-1: {4:8.4f}".format(
                    epoch, train_loss[epoch], valid_loss[epoch], train_top1_acc[epoch], valid_top1_acc[epoch]))
        else:
            down_rounds += 1

        if down_rounds >= patience:
            print("Earlying stopping at epoch {}.".format(epoch))
            break

    return train_loss, valid_loss, train_top1_acc, valid_top1_acc, train_top5_acc, valid_top5_acc
