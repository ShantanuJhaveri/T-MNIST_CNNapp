import numpy as np
import cv2 as cv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


class NN(object):
    def __init__(self):
        # load in our training and testing data from MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # feature extraction, scaling, and normalization
        # could have used the TTsplit but trying to stay on purely KERAS
        # KERAS API requires (num of inputs, rows, columns, dimension - [1 for gray 3 for rgb])
        self.train_im = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
        # the categorical for the y basically converts 0 -> [1x10 array] and then 1 -> [1x10 array]
        self.train_tar = to_categorical(y_train)
        self.test_im = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
        # the categorical for the y basically converts 0 -> [1x10 array] and then 1 -> [1x10 array]
        self.test_tar = to_categorical(y_test)
        self.in_shape = (self.train_im.shape[1], 1)

        # THIS IS ALL THE FUCKING MODEL...? THATS IT.
        # A Sequential model is appropriate for a plain stack of layers where each
        # layer has exactly one input tensor and one output tensor.
        # there are also functional models that we will go over later
        self.model = Sequential()

        # Adding each layer by layer with .add
        # but the things needed for the .add are (num_filters, filter_size, pool_size)
        # I also added our AF and input shape to validate, but not needed
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))


        self.model.add(Flatten())

        # DENSE LAYER TO CONVERT TO 10 CLASS DENSE MAP
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        # bro is that really that is needed for the backprop????
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 50 epochs / test train validation split .3
        self.model.fit(self.train_im, self.train_tar, validation_split=0.3,
                       callbacks=[EarlyStopping(patience=2)], epochs=3)
        # callback will allow us to periodically save our model throughout training just in case

        # Predict with testing data on the first 5 images
        # predictions = self.model.predict(X_test[:5])
        # print(np.argmax(predictions, axis=1))
        # print(y_test[:5])

    # run our live prediction inputs instead of just running our testing data
    def predict(self, image):
        input = cv.resize(image, (28, 28)).reshape((28, 28, 1)).astype('float32') / 255
        return self.model.predict_classes(np.array([input]))
