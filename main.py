"""
Train a convolutional neural network
using the Keras framework.

David Riser
April 20, 2020
"""

import argparse
import numpy as np
import os
import pandas as pd

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     Dense, Softmax, Input, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_dataset(args):
    """ Download and return either CIFAR-10
    or CIFAR-100.
    """
    if args.dataset == "cifar10":
        return cifar10.load_data()
    elif args.dataset == "cifar100":
        return cifar100.load_data(label_mode='fine')

def conv_block(inputs, nfilters, filter_size, strides):
    """ Build the pattern of C1D, Act, BN. """
    x = Conv2D(nfilters, filter_size, strides)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    return x

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

def build_cnn(input_shape, output_shape):
    input_layer = Input(shape=input_shape)

    x = conv_block(input_layer, 16, (3, 3), (1, 1))
    x = conv_block(x, 32, (3, 3), (1, 1))
    x = MaxPooling2D()(x)

    x = conv_block(x, 32, (3, 3), (1, 1))
    x = conv_block(x, 64, (3, 3), (1, 1))
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    # x = Dense(256)(x)
    # x = Activation('relu')(x)
    x = Dense(output_shape)(x)
    x = Activation('relu')(x)
    output_layer = Softmax()(x)

    model = Model(input_layer, output_layer)
    return model

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Train a CNN using Keras on CIFAR-10/CIFAR-100')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batches_per_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_decreases', type=int, default=3)
    args = parser.parse_args()

    # Create filesystem
    project_dir = os.path.join(args.output_dir, args.experiment)
    make_directory(args.output_dir)
    make_directory(project_dir)
    make_directory(project_dir + '/models')
    make_directory(project_dir + '/metrics')
    print('[INFO] Project structure created.')
    
    # Download data set and convert the categorical values into
    # label vectors.  Normalize the images.
    (X_train, Y_train), (X_test, Y_test) = build_dataset(args)
    Y_train = to_categorical(Y_train)
    X_train = X_train / 255
    X_test = X_test / 255

    # Build the data generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.1
    )
    print('[INFO] Data generator built.')
    
    # Train with a simple stepwise
    # learning rate schedule.
    learning_rate_schedule = []
    for i in range(args.learning_rate_decreases):
        factor = 10. * i if i > 0 else 1.
        learning_rate_schedule.append(args.learning_rate / factor)

    print('[INFO] Learning schedule {}'.format(learning_rate_schedule))
    
    # Build model
    model = build_cnn(X_train.shape[1:], Y_train.shape[1])
    print('[INFO] Model built')
    print(model.summary())
    
    t_loss = []
    v_loss = []
    t_top1 = []
    v_top1 = []
    t_top5 = []
    v_top5 = []
    for learning_rate in learning_rate_schedule:
        print('[INFO] On learning rate step: {}'.format(learning_rate))
        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        tl, vl, tt1, vt1, tt5, vt5 = train_model_with_augmentations(
            model=model,
            datagen=datagen,
            X_test=X_test,
            Y_test=Y_test,
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            patience=args.patience,
            savename='{}/models/weights.{}.h5'.format(project_dir, args.dataset)
        )

        # Add to the bigger list
        t_loss.extend(tl)
        v_loss.extend(vl)
        t_top1.extend(tt1)
        v_top1.extend(vt1)
        t_top5.extend(tt5)
        v_top5.extend(vt5)

    data = {
        'train_loss':t_loss,
        'valid_loss':v_loss,
        'train_top1':t_top1,
        'valid_top1':v_top1,
        'train_top5':t_top5,
        'valid_top5':v_top5
    }

    # Save metrics to output
    df = pd.DataFrame(data)
    df.to_csv('{}/metrics/metrics.csv'.format(project_dir), index=False)

    print('[INFO] Finished! ')
