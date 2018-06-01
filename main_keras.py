from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from main import *
import keras

batch_size = 128
img_rows = 28
img_cols = 28

x_train, y_train, x_test, y_test, val, val_label = get_stratified_data()

epochs=  20
num_classes = len(np.unique(val_label))

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    val = val.reshape(val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    val = val.reshape(val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
val_label = keras.utils.to_categorical(val_label, num_classes)

def add_conv_batch_block(model, num_filters, kernel_size):
    model.add(Conv2D(num_filters, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape))
add_conv_batch_block(model, 32, (3,3))
add_conv_batch_block(model, 64, (3,3))
add_conv_batch_block(model, 128, (3,3))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val, val_label))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
