#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 00:50:59 2018

@author: optnio
"""
from model_object_detection import Model_Object_Detection

model = Model_Object_Detection("Object Detection", output_folder="output")
model.init()