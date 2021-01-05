import sys
from time import sleep

from pygame.tests.draw_test import RED

from activations.ReLU import ReLU
from activations.Sigmoid import Sigmoid
from activations.SoftMax import SoftMax
from activations.Tanh import Tanh
from enums.TypeOfLayer import TypeOfLayer
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from loss.MeanSquaredError import MeanSquaredError
from metrics.Accuracy import Accuracy
from metrics.CustomAccuracy import CustomAccuracy
from models import NeuralStatistics
from models.Layer import Layer
from models.NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import mnist
from matplotlib import pyplot
import keras
import keras.utils
from keras import utils as np_utils
from tensorflow.keras import utils as np_utils
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY


train_X, train_Y, test_X, test_Y = load_dataset()

# shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_Y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_Y.shape))

# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

train_norm = train_X.astype('float32')
test_norm = test_X.astype('float32')
# normalize to range 0-1
train_norm_X = train_norm / 255.0
test_norm_X = test_norm / 255.0

from activations.SoftMax import SoftMax
from metrics.Accuracy import Accuracy

model = NeuralNetwork()
model.add_layer(Layer(units=784, type=TypeOfLayer.INPUT))  # 0
model.add_layer(Layer(units=40, activation=ReLU(), type=TypeOfLayer.HIDDEN))  # 1
model.add_layer(Layer(units=10, activation=SoftMax(), type=TypeOfLayer.OUTPUT))  # 2

model.compile(loss=MeanSquaredError(), metrics=[CustomAccuracy()])


train_X = train_X.reshape((train_norm_X.shape[0], 28*28))
test_X = test_X.reshape((test_norm_X.shape[0], 28*28))
model.fit(x=train_X, y=train_Y, epochs=10, learning_rate=0.01)
