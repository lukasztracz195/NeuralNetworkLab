from activations.Activation import Activation
import numpy as np


class Sigmoid(Activation):
    def count_value(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def count_value_of_derivative(self, x):

        return x * (1-x)