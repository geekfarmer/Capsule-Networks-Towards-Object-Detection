#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:23:02 2018

@author: optnio
"""

import os
import pickle

TRAIN_FILE = "train"
VALID_FILE = "train"
TEST_FILE = "test"

def get_data(folder):

    # Load the dataset
    training_file = os.path.join(folder, TRAIN_FILE)
    validation_file= os.path.join(folder, VALID_FILE)
    testing_file =  os.path.join(folder, TEST_FILE)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Retrive all datas
    x_train, y_train = train['features'], train['labels']
    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']

    return x_train, y_train, x_valid, y_valid, x_test, y_test