from activations.Activation import Activation
import numpy as np


class SoftMax(Activation):
    def count_value(self, x):
        """Compute the softmax of vector x."""
        return np.exp(x)/sum(np.exp(x))

    def count_value_of_derivative(self, x):
        jacobian_m = np.diag(x)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = x[i] * (1 - x[i])
                else:
                    jacobian_m[i][j] = -x[i] * x[j]
        return jacobian_m
