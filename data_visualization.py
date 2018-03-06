#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:03:57 2018

@author: optnio
"""

import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance
# Visualizations will be shown in the notebook.
from load_data import x_train,y_train,x_test,y_test,n_classes

# Load name of id
with open("./data/classnames.csv", "r") as f:
    classnames = f.read()
id_to_name = { int(line.split(",")[0]):line.split(",")[1] for line in classnames.split("\n")[1:] if len(line) > 0}


graph_size = 3
random_index_list = [random.randint(0, x_train.shape[0]) for _ in range(graph_size * graph_size)]
fig = plt.figure(figsize=(15, 15))
"""for i, index in enumerate(random_index_list):
    a=fig.add_subplot(graph_size, graph_size, i+1)
    #im = Image.fromarray(np.rollaxis(X_train[index] * 255, 0,3))
    imgplot = plt.imshow(x_train[index])
    # Plot some images
    a.set_title('%s' % id_to_name[y_train[index]])

#plt.show()

"""

fig, ax = plt.subplots()
# the histogram of the data
values, bins, patches = ax.hist(y_train, n_classes, normed=10)

# add a 'best fit' line
ax.set_xlabel('Smarts')
ax.set_title('Histogram of classess')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

print ("Most common index")
most_common_index = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
for index in most_common_index[:30]:
    print("index: %s => %s = %s" % (index, id_to_name[index], values[index]))

