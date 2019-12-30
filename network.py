# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle

CLASS_NAMES = ['Cover', 'Stego']
NUM_COLS = 5
NUM_ROWS = 5

def normalize(arr: np.ndarray):
    mean = np.mean(arr, dtype=np.float64)
    std = np.std(arr, dtype=np.float64)
    arr = arr[:][:] * mean / std

def prepare_data(array):
    array = array / 255.0

    normalize(array)

    return np.expand_dims(array, axis=3)

class Network(object):
    def __init__(self):
        self.model = keras.Sequential([
            Conv2D(filters=4, kernel_size=(3, 3),
                                input_shape=(32, 32, 1), activation='tanh', kernel_initializer='glorot_uniform'),
            Conv2D(filters=256, kernel_size=(29, 29), activation='tanh', kernel_initializer='glorot_uniform'),
            Flatten(),
            Dense(256 * 4, kernel_initializer='glorot_uniform', activation='tanh'),
            Dense(2, activation='softmax',
                kernel_initializer='glorot_uniform')
        ])

        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=3e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    def load_model(self, path):
        print('Started loading model: {}'.format(path))
        self.model = keras.models.load_model(path)
        print('Finished loading model!\n')
    
    def load_training_data(self, path):
        print('Started loading training data: {}'.format(path))
        (train_images, train_labels), (test_images, test_labels) = pickle.load(open(path, "rb"))
        train_images, train_labels = shuffle(train_images, train_labels)

        print('Started preparing data.')
        train_images = prepare_data(train_images)
        test_images = prepare_data(test_images)

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        print('Finished loading training data!\n')
    
    def model_description(self):
        self.model.summary()
    
    def train(self, epochs):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, validation_data=(self.test_images, self.test_labels))
    
    def save_model(self, filename):
        self.model.save_weights("models/{}.h5".format(filename))
        print('File saved: {}.h5'.format(filename))
    
    def evaluate(self, data):
        data = prepare_data(data)
        predictions = self.model.predict_classes(data)
        probabilities = self.model.predict(data)

        count = len(data) if len(data) < NUM_ROWS * NUM_COLS else NUM_ROWS * NUM_COLS
        plt.figure(figsize=(10, 10))
        for i in range(count):
            plt.subplot(NUM_ROWS, NUM_COLS, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(data.squeeze(3)[i], cmap='gray')
            plt.xlabel('{} ({}% sure)'.format(CLASS_NAMES[predictions[i]], str(probabilities[i][predictions[i]] * 100)[:4]))
        plt.show()