from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     Dense, Softmax, Input, Flatten,
                                     MaxPooling2D, Dropout)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def conv_block(inputs, nfilters, filter_size, strides, use_batchnorm=True):
    x = Conv2D(nfilters, filter_size, strides)(inputs)
    x = Activation('relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    return x

def init_model(params):
    """ Params dictionary based model builder. """
    
    input_layer = Input(shape=params['input_shape'])

    x = conv_block(input_layer, params['init_filters'], (3, 3), (1, 1), params['use_batchnorm'])
    x = conv_block(x, params['init_filters'] * 2, (3, 3), (1, 1), params['use_batchnorm'])
    x = MaxPooling2D()(x)

    if params['depth'] > 1:
        filter_mult = 2
        for depth in range(2,params['depth']):
            x = conv_block(x, params['init_filters'] * filter_mult, (3, 3), (1, 1), params['use_batchnorm'])
            filter_mult += 1
            x = conv_block(x, params['init_filters'] * filter_mult, (3, 3), (1, 1), params['use_batchnorm'])
            x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(params['output_shape'])(x)
    x = Activation('relu')(x)
    output_layer = Softmax()(x)

    model = Model(input_layer, output_layer)
    return model

def init_adam(params):
    adam = Adam(params['learning_rate'], params['beta1'], params['beta2'])
    return adam
