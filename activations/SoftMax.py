from sympy.codegen.ast import float128

from activations.Activation import Activation
import numpy as np

import numpy as np


class SoftMax(Activation):
    def count_value(self, x):
        """Compute the softmax of vector x."""
        np.seterr(divide='ignore', invalid='ignore')
        value = np.exp(x) / sum(np.exp(x))
        return value

    def count_value_of_derivative(self, x):
        pass

    # jacobian_m = np.diag(x)
    # for i in range(len(jacobian_m)):
    #     for j in range(len(jacobian_m)):
    #         if i == j:
    #             jacobian_m[i][j] = x[i] * (1 - x[i])
    #         else:
    #             jacobian_m[i][j] = -x[i] * x[j]
    # return jacobian_m
