# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import cv2
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
#from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import numpy as np

# read data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# limit the amount of the data
# train data
ind_train = random.sample(list(range(x_train.shape[0])), 50000)
x_train = x_train[ind_train]
y_train = y_train[ind_train]

# test data
ind_test = random.sample(list(range(x_test.shape[0])), 10000)
x_test = x_test[ind_test]
y_test = y_test[ind_test]
"""
def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 28, 28, 1))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled


# resize train and  test data
x_train_resized = resize_data(x_train)
x_test_resized = resize_data(x_test)

# make explained variable hot-encoded
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_test)

"""
# TODO: Number of training example
n_train = x_train.shape[0]

# TODO: Number of testing example.
n_test = x_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = x_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
#n_classes = len(set(y_train))
n_classes = 100
"""print(x_train.shape)
print(y_train.shape)



print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

"""

