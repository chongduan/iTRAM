# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
#import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Avoid warning re Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load simulation data
path = '/mnt/hgfs/ChongDuan/13_iTRAM/Simulation/iTRAM-T1T2.mat'

X = loadmat(path)['X']
y = loadmat(path)['y']


# Split data into Train, Test, and Validation
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=100)

# Build model
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Split training data to get validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.1, random_state=100)

history = model.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val))

# list all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# apply to test data
model.evaluate(X_test, y_test)