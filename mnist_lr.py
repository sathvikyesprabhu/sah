'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, sgd 
import matplotlib.pyplot as plot 

batch_size = 32
num_classes = 10
epochs = 25

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# DISPLAY SOME DATA
for i in range(12):
    plot.subplot(4,3,i+1)
    x = x_train[i,:].reshape((28,28));
    plot.imshow(x); 
    plot.axis('off');
    plot.title(y_train[i]);
   
plot.show();


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(num_classes, activation='softmax', input_shape=(784,)))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=sgd(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


W = model.layers[0].get_weights(); 
wts = W[0]; 

print(wts.shape);

for i in range(10):
    plot.subplot(4,3,i+1)
    x = wts[:,i].reshape((28,28));
    plot.imshow(x); 
    plot.axis('off');
    plot.title(i);
   
plot.show();


