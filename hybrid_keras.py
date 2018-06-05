import glob
import shutil
import uuid
import math
import sys
import numpy as np
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Input
from keras.layers.merge import Concatenate, Average
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.metrics as metrics
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def inception_module(X, scale=1, predict=False):
    x11 = Conv2D(int(16*scale), (1, 1), padding='valid')(X)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)

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

    if predict:
        predict = AveragePooling2D((5, 5), strides=(1, 1))(X)
        predict = Conv2D(int(8*scale), (1, 1))(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dropout(0.25)(predict)
        predict = Flatten()(predict)
        predict = Dense(120)(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dense(num_classes, activation='softmax')(predict)
        return out, predict

    return out

def inception_model(X):
    x = Reshape((img_cols, img_rows, 1))(X)
    x = inception_module(x, 1)
    x = Conv2D(16, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = inception_module(x, 2)
    x = inception_module(x, 3)

    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)

    out = Activation('softmax')(x)
    #soft3 = Activation('softmax')(x)
    #out = Average()([soft1, soft2, soft3])

    return out

def build_network():
    x = Input((img_rows*img_cols,))
    incep_n = inception_model(x)
    model = Model(inputs=x, outputs=[incep_n])

    return model


def get_data():
    x_train = np.load('train.npy')
    y_train = np.load('train_labels.npy')

    val = np.load('vali.npy')
    val_label = np.load('vali_labels.npy')

    global img_rows
    global img_cols

    img_rows = int(math.sqrt(x_train.shape[1]))
    img_cols = img_rows

    global num_classes
    num_classes = len(np.unique(val_label))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    val_label = keras.utils.to_categorical(val_label, num_classes)
    return x_train, y_train, val, val_label

def train():
    x_train, y_train, val, val_label = get_data()

    model = build_network()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    name = str(uuid.uuid4())
    save_best_model = ModelCheckpoint("models/keras/"+name+".h5",
                                      monitor='val_acc', verbose=1,
                                      save_best_only=True, mode='max',
                                      save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.05,
                                   patience=3, verbose=1)
    model.fit(x_train, y_train, batch_size=128,
              epochs=200, validation_data=(val, val_label)
             ,verbose=1, callbacks=[save_best_model, early_stopping])

    model.load_weights("models/keras/"+name+".h5")
    score = model.evaluate(val, val_label)
    shutil.move("models/keras/"+name+".h5", "models/keras/"+str((1.-score[1])*100)+".h5")
    print('--------------------------------------')
    print('model'+str(name)+':')
    print('Test loss:', score[0])
    print('error:', str((1.-score[1])*100)+'%')

    return score

def pred_ensemble():

    _, _, val, val_label = get_data()

    ensemble = []
    for weights in glob.glob("models/keras/*.h5"):
        model = build_network()
        model.load_weights(weights)
        ensemble.append(model)
    x = Input((img_rows*img_cols,))
    outputs = [model(x) for model in ensemble]
    y = Average()(outputs)

    ensemble_model = Model(x, y)

    preds = ensemble_model.predict(val)

    preds = np.argmax(preds, axis=1)
    print(preds)
    print(val_labels)
    metrics = [accuracy_score, recall_score, precision_score, f1_score]

    for metric in metrics:
        if str(metric.__name__) == "accuracy_score":
            print(metric.__name__,":",metric(np.array(val_label), preds))
        else:
            print(metric.__name__,":",metric(np.array(val_label), preds, average='macro'))

def main(ensemble=False):
    if ensemble:
        pred_ensemble()
    else:
        train()


if __name__== "__main__":
    train()
    #if sys.argv == "TRAIN":
#        main(False)
#    else:
#        main(True)
