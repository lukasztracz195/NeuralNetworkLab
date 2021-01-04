from activations.Activation import Activation
import numpy as np


class ReLU(Activation):

    def count_value(self, x):
        return np.maximum(x, 0)

    def count_value_of_derivative(self, x):
        x_tmp = x.copy()
        x_tmp[x_tmp <= 0] = 0
        x_tmp[x_tmp > 0] = 1
        return x_tmp
