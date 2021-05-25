import numpy as np
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from PIL import Image


def print_prediction(model):
    im = Image.open('five.png')
    im = im.convert('L').resize((28, 28))
    im = np.array(im, dtype='int32')
    im = np.reshape(im, (1, 784))

    prediction = model.predict_classes(im)[0]
    print('prediction:', prediction)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)

no_classes = 10
y_train = np_utils.to_categorical(y_train, no_classes)
y_test = np_utils.to_categorical(y_test, no_classes)

model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
'''

model.save('digits_rec_model.h5')
