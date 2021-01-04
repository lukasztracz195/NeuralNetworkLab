from activations.Activation import Activation
import numpy as np


class Tanh(Activation):
    def count_value(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def count_value_of_derivative(self, x):
        return 1 - x ** 2
