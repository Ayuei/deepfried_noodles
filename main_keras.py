from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
import numpy as np
import math

batch_size = 512

x_train = np.load('train.npy')
y_train = np.load('train_labels.npy')
val = np.load('vali.npy')
val_label = np.load('vali_labels.npy')

img_rows = int(math.sqrt(x_train.shape[1]))
img_cols = img_rows

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
val = val.reshape(val.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

epochs = 200
num_classes = len(np.unique(val_label))

y_train = keras.utils.to_categorical(y_train, num_classes)
val_label = keras.utils.to_categorical(val_label, num_classes)

def add_conv_batch_block(model, num_filters, kernel_size, use_pooling=False):

    model.add(Conv2D(num_filters, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if use_pooling:
        model.add(MaxPooling2D(2, padding='same'))

model = Sequential()
#model.add(Reshape((img_rows, img_cols, 1), input_shape=(img_rows*img_cols,)))
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
add_conv_batch_block(model, 64, (3,3), True)
add_conv_batch_block(model, 128, (3,3), True)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
#image_gen = ImageDataGenerator(rotation_range=45,
#                               width_shift_range=0.10,
#                               height_shift_range=0.10,
#                               horizontal_flip=True,
#                               vertical_flip=True)
#image_gen.fit(x_train)
#model.fit_generator(image_gen.flow(x_train, y_train, batch_size=batch_size),
#                    epochs=epochs, validation_data=(val,val_label))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val, val_label))

score = model.evaluate(val, val_label, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

model.save(str(score[1]).replace(".", "_")+".h5")
print("Modelled saved as:", str(score[1]).replace(".", "_")+".h5")
