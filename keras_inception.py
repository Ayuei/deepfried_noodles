import os
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.merge import Concatenate, Average
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.metrics as metrics

def predict_layer(X, scale):
    predict = AveragePooling2D((5, 5), strides=(1, 1))(x)
    predict = Conv2D(int(8*scale), (1, 1))(predict)
    predict = BatchNormalization()(predict)
    predict = Activation('relu')(predict)
    predict = Dropout(0.25)(predict)
    predict = Flatten()(predict)
    predict = Dense(120)(predict)
    predict = BatchNormalization()(predict)
    predict = Activation('relu')(predict)
    predict = Dense(10, activation='softmax')(predict)

    return out, predict


def inception_module(X, scale_factor=1, predict=False):
    x1_1 = Conv2D(int(16*scale), (1,1), padding='valid')(X)
    x1_1 = BatchNormalization()(X1_1)
    x1_1 = Conv2D(int(16*scale), (1,1), padding='valid')(X1_1)
    x1_1 = Activation('relu')(x1_1)

    x33 = Conv2D(int(24*scale), (1, 1))(X)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)
    x33 = Conv2D(int(32*scale), (3, 3), padding='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)

    x55 = Conv2D(int(4*scale), (1, 1))(X)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)
    x55 = Conv2D(int(8*scale), (5, 5), padding='same')(x55)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)

    x33p = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
    x33p = Conv2D(int(8*scale), (1, 1))(x33p)
    x33p = BatchNormalization()(x33p)
    x33p = Activation('relu')(x33p)

    out = Concatenate(axis=3)([x11, x33, x55, x33p])

    return out

def inception_model(x):
    x = Conv2D(16, (3, 3),strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(48, (3, 3),strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = inception_module(x, 1)
    x = inception_module(x, 2)
    x = inception_module(x, 2)
    x, soft1 = inception_module(x, 3)

    soft1 = predict(x)

    x = inception_module(x, 3)
    x = inception_module(x, 3)
    x = inception_module(x, 4)

    soft2 = predict(x)

    x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = inception_module(x, 4)
    x = inception_module(x, 5)
    x = AveragePooling2D((5, 5), strides=(1, 1))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    soft3 = Activation('softmax')(x)
    out = Average()([soft1, soft2, soft3])

    return out
