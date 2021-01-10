from sympy.codegen.ast import float128

from activations.Activation import Activation
import numpy as np

import numpy as np


class Sigmoid(Activation):
    def count_value(self, x):
        value = 1.0 / (1.0 + np.exp(-x))
        return value

    def count_value_of_derivative(self, x):
        value = x * (1 - x)
        return value
