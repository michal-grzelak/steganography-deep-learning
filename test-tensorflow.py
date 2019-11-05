from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Reshape

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
from shutil import copyfile
import collections
import random

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

# arrays specifying folders for images and labels for each folder
# trains_paths = [
#     ["C:\Development\Coding\steganography-deep-learning\\raise\cover_pgm\\train\\", 0],
#     ["C:\Development\Coding\steganography-deep-learning\\raise\stego_hugo_0.4\\train\\", 1]
# ]


# test_paths = [
#     ["C:\Development\Coding\steganography-deep-learning\\raise\cover_pgm\\test\\", 0],
#     ["C:\Development\Coding\steganography-deep-learning\\raise\stego_hugo_0.4\\test\\", 1]
# ]
trains_paths = [
    ["C:\Development\Coding\steganography-deep-learning\Corel10k\cover_pgm\\126x187\\train\\", 0],
    ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_uniward_0.4\\126x187\\train\\", 1],
    # ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_wow_0.4\\126x187\\train\\", 1],
    # ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_uniward_0.4\\126x187\\train\\", 1],
]

test_paths = [
    ["C:\Development\Coding\steganography-deep-learning\Corel10k\cover_pgm\\126x187\\test\\", 0],
    ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_uniward_0.4\\126x187\\test\\", 1],
    # ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_wow_0.4\\126x187\\test\\", 1],
    # ["C:\Development\Coding\steganography-deep-learning\Corel10k\stego_uniward_0.4\\126x187\\test\\", 1],
]

# data loader, return data in format (images, labels),
# where images is 2d numpy array and labels is 1d numpy array


def load_data(paths):
    images = []
    labels = np.empty(0, dtype=int)

    for path in paths:
        seti = []
        files = get_files_in_directory(path[0])

        for file in files:
            try:
                seti.append(read_pgm(file))
            except:
                print(file)

        labels = np.concatenate(
            (labels, np.full(len(seti), path[1], dtype=int)), axis=0)
        images = np.concatenate((images, np.array(seti)),
                                axis=0) if images != [] else np.array(seti)

    return (np.array(images), labels)

def set_seed(seed): 
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)


def train():
    # load train and test data
    (train_images, train_labels) = load_data(trains_paths)
    (test_images, test_labels) = load_data(test_paths)

    class_names = ['cover', 'stego']

    # load mnist
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()



    # scale values to a range 0 -> 1
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0

    # random indice shuffle for test image display
    # indices = np.arange(train_labels.shape[0])
    # np.random.shuffle(indices)

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[indices[i]], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[indices[i]]])
    # plt.show()

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # test_dataset = test_dataset.batch(BATCH_SIZE)



    input_shape = train_images.shape[1:]
    
    model = keras.Sequential([
        Conv2D(filters=1, kernel_size=(3, 3),
                            input_shape=input_shape, activation='tanh'),
        Conv2D(filters=64, kernel_size=tuple(np.subtract(input_shape[:-1], (3, 3))), activation='tanh'),
        Reshape((64, 2, 2)),
        Dense(64, input_shape=(2, 2)),
        Flatten(),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0008),
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
    model.load_weights("init1.h5")
    history = model.fit(train_images, train_labels, epochs=150, callbacks=[cp_callback], validation_data=(test_images,test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    model.save('model-best.h5') 


    # plot train accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1.0])
    plt.legend(loc='lower right')
    plt.show()


    print('\nTest accuracy:', test_acc)


with tf.device('/device:gpu:0'):
    train()
