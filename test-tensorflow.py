from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
from shutil import copyfile
import collections
import random
import timeit
import statistics
import pickle
from sklearn.utils import shuffle


# pgm images reader
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


# get all files in specified directory
def get_files_in_directory(dir, format='*', recursive=False):
    file_regex = '*.' + format
    files = file_regex if not recursive else '**\\' + file_regex
    return [f for f in glob.glob(dir + files, recursive=recursive)]

# data loader, return data in format (images, labels),
# where images is 2d numpy array and labels is 1d numpy array


def load_data(paths):
    images = []
    labels = np.empty(0, dtype=int)

    for path in paths:
        seti = []
        files = get_files_in_directory(path[0])

        for i in range(len(files)):
            try:
                image = read_pgm(files[i])
                seti.append(image)
            except:
                print(files[i])

        labels = np.concatenate(
            (labels, np.full(len(seti), path[1], dtype=int)), axis=0)
        images = np.concatenate((images, np.array(seti)),
                                axis=0) if images != [] else np.array(seti)

    return (np.array(images), labels)


def set_seed(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def model1(input_shape, class_names):
    return keras.Sequential([
        Conv2D(filters=32, kernel_size=(5, 5),
               input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='tanh',
               kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='tanh',
               kernel_initializer='glorot_uniform'),
        # MaxPooling2D((2, 2)),
        # Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
        #        kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='tanh', kernel_initializer='glorot_uniform'),
        Dense(256, activation='tanh', kernel_initializer='glorot_uniform'),
        Dense(512, activation='tanh', kernel_initializer='glorot_uniform'),
        Dense(64, activation='tanh', kernel_initializer='glorot_uniform'),
        Dense(len(class_names), activation='softmax',
              kernel_initializer='glorot_uniform')
    ])

def model3(input_shape, class_names):
    return keras.Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), padding='same',
               input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
               kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
               kernel_initializer='glorot_uniform'),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
               kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
               kernel_initializer='glorot_uniform'),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
               kernel_initializer='glorot_uniform'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(4096, activation='relu', kernel_initializer='glorot_uniform'),
        Dense(4096, activation='relu', kernel_initializer='glorot_uniform'),
        Dense(1000, activation='relu', kernel_initializer='glorot_uniform'),
        Dense(len(class_names), activation='softmax',
              kernel_initializer='glorot_uniform')
    ])


def model2(input_shape, class_names):
    return keras.Sequential([
        Conv2D(filters=1, kernel_size=(3, 3),
                            input_shape=input_shape, activation='tanh', kernel_initializer='random_normal', bias_initializer='random_normal'),
        Conv2D(filters=64, kernel_size=tuple(np.subtract(
            input_shape[:-1], (3, 3))), activation='tanh', kernel_initializer='random_normal', bias_initializer='random_normal'),
        Flatten(),
        Dense(64 * 4, kernel_initializer='random_normal', activation='tanh', bias_initializer='random_normal'),
        Dense(len(class_names), activation='softmax',
              kernel_initializer='random_normal', bias_initializer='random_normal')
    ])

def normalize(arr: np.ndarray):
    mean = np.mean(arr, dtype=np.float64)
    std = np.std(arr, dtype=np.float64)
    arr = arr[:][:] * mean / std


def train(train_images, train_labels, test_images, test_labels, class_names, learning_rate, save_name, epochs):
    input_shape = train_images.shape[1:]

    model = model2(input_shape, class_names)

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    load_path = "training_0_74/cp.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # model.load_weights(load_path)
    # model.load_weights("init1.h5")
    model.save_weights("{}.h5".format(save_name))
    class_weight = {0: 0.5, 1: 0.5}

    BATCH_SIZE = 100
    SHUFFLE_BUFFER_SIZE = 100

    train_images, train_labels = shuffle(train_images, train_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    history = model.fit(train_dataset, epochs=epochs, callbacks=[
                        cp_callback], validation_data=test_dataset, class_weight=class_weight)

    W = model.get_weights()

    model.save("{}-{}.h5".format(save_name, "trained"))

    predictions = model.predict_classes(test_images)
    cm = tf.math.confusion_matrix(test_labels, predictions)

    # FalsePositive = []
    # for i in range(2):
    #     FalsePositive.append(sum(cm[:,i]) - cm[i,i])
    # FalseNegative = []
    # for i in range(2):
    #     FalseNegative.append(sum(cm[i,:]) - cm[i,i])
    print(predictions)
    print(cm)
    # print(FalseNegative)
    # print(FalsePositive)

    return model, history


def findb():
    (train_images, train_labels), (test_images, test_labels) = pickle.load(
        open("data/32x32/data_small.pkl", "rb"))
    
    # (train_images2, train_labels2), (test_images2, test_labels2) = pickle.load(
    #     open("data/32x32/data_small_noise_10-10.pkl", "rb"))

    class_names = ['cover', 'stego']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    normalize(train_images)
    normalize(test_images)

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # test_images2 = test_images2 / 255.0
    # normalize(test_images2)
    # test_images2 = np.expand_dims(test_images2, axis=3)

    # model, history = train(train_images, train_labels, test_images, test_labels, class_names, 0.001, "find/{}".format("init"), 1000)
    model, history = train(train_images, train_labels, test_images,
                           test_labels, class_names, 1e-4, "find/{}".format("init"), 20)
    model.save_weights("trained-1000.h5")

    # plot train accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1.0])
    plt.legend(loc='lower right')
    plt.show()

    # for i in range(100):
    #     model, history = train(train_images, train_labels, test_images, test_labels, class_names, 0.00008, "find/{}".format(i))
    #     test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #     if history.history['accuracy'][-1] > 0.58:
    #         print("Found weights that goes >0.58 for i: {}".format(i))
    #         model.save("{}-{}.h5".format("find/found/{}".format(i), "trained"))


with tf.device('/device:gpu:0'):
    findb()
