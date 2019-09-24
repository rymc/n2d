import os

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y


def load_mnist_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test
    y = y_test
    x = np.divide(x, 255.)
    x = x.reshape((x.shape[0], -1))
    return x, y


def load_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names


def load_har():
    x_train = pd.read_csv(
        'data/har/train/X_train.txt',
        sep=r'\s+',
        header=None)
    y_train = pd.read_csv('data/har/train/y_train.txt', header=None)
    x_test = pd.read_csv('data/har/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('data/har/test/y_test.txt', header=None)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    y_names = {0: 'Walking', 1: 'Upstairs', 2: 'Downstairs', 3: 'Sitting', 4: 'Standing', 5: 'Laying', }
    return x, y, y_names


def load_usps(data_path='data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_pendigits(data_path='data/pendigits'):
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.makedirs(data_path,  exist_ok=True)
        
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' %
            data_path)
        os.system(
                'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' %
            data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')
    return x, y
