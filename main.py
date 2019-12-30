from network import Network
from menu import Menu
import glob
import os
from PIL import Image
import numpy as np
from input_provider import InputProvider
import sys

def get_files_in_directory(dir, format='*', recursive=False):
    file_regex = '*.' + format
    files = file_regex if not recursive else '**\\' + file_regex
    return [f for f in glob.glob(os.path.join(dir, files), recursive=recursive)]

def load_image_asarray(path):
    return np.asarray(Image.open(path).convert('L').resize((32, 32), Image.LANCZOS))

def train(selected: int):
    files = get_files_in_directory('train_data/', 'pkl')
    file_menu = Menu(files, lambda x: network.load_training_data(files[x - 1]), 'Choose train data:', has_back=True)
    file_menu.display()
    inp = file_menu.select()
    if inp == None:
        return

    epochs = int(InputProvider('Enter number of epochs: ', int, True).input())
    network.train(epochs)

    yesno_menu = Menu([
        'Yes',
        'No'
    ], [
        lambda x: network.save_model(input('Enter filename: ')),
        lambda x: 0
    ], 'Save model?')
    yesno_menu.display()
    yesno_menu.select()

def load_model(selected: int):
    models = get_files_in_directory('models/', 'h5')
    model_menu = Menu(models, lambda x: network.load_model(models[x - 1]), 'Choose model:', has_back=True)
    model_menu.display()
    model_menu.select()

def evaluate(selected: int):
    path = input('Provide directory or filename (*.jpg, *.png, etc.): ')

    images = []
    try:
        if os.path.isfile(path):
            images = np.expand_dims(load_image_asarray(path), axis=0)
        elif os.path.isdir(path):
            files = get_files_in_directory(path)
            if len(files) < 1:
                raise IndexError()

            for file in files[0:25]:
                images.append(load_image_asarray(file))
            images = np.array(images)
        else:
            raise IOError()
        
        network.evaluate(images)
    except IOError:
        print('Error! Valid directory or file must be provided!\n')
    except IndexError:
        print('Directory is empty!\n')

main_menu = Menu([
    'Train',
    'Evaluate',
    'Load saved model from file',
    'Exit'
], [
    train,
    evaluate,
    load_model,
    lambda x: sys.exit()
])

network = Network()

while True:
    try:
        main_menu.display()
        opt = main_menu.select()
    except SystemExit:
        print('Exiting...')
        break
    except RuntimeError:
        print('Unidentified error! Exiting!')
        break
