import numpy as np

from enums.TypeOfLayer import TypeOfLayer
from models.Layer import Layer


class Dense(Layer):

    def __init__(self, units: int, weight_min_value=0.0, weight_max_value=1.0, activation=None,
                 type=TypeOfLayer.UNDEFINED):
        super().__init__(units, weight_min_value, weight_max_value, activation, type)

    def forward(self, input_value):
        if self.previously_layer is not None:
            if  self.previously_layer.type == TypeOfLayer.INPUT:
                weights = self.weights.T
                self.values = self.multiply(input_value, weights)
                if self.type == TypeOfLayer.HIDDEN and self.activation is not None:
                    self.values = self.activation.count_value(self.values)
            else:
                values = self.previously_layer.values
                weights = self.weights.T
                self.values = self.multiply(values, weights)
                if self.activation is not None:
                    self.values = self.activation.count_value(self.values)

    def backward(self, expected_value: float, input_value):
        self.__count_delta(expected_value=expected_value)
        self.__count_weight_delta(input_value=input_value)

    def change_weights(self, learning_rate: float):
        self.__change_weights(learning_rate=learning_rate)

    def __count_delta(self, expected_value):
        if self.type == TypeOfLayer.OUTPUT:
            values = self.values
            self.delta = values - expected_value
        else:
            if self.next_layer is not None:
                delta_from_previously_layer = self.next_layer.delta
                weights = self.next_layer.weights
                self.delta = self.multiply(delta_from_previously_layer, weights)
                if self.activation is not None:
                    derivative_values = self.activation.count_value_of_derivative(self.values)
                else:
                    derivative_values = 1.0
                self.delta = self.delta * derivative_values

    def __count_weight_delta(self, input_value):
        weight_delta = 0.0
        if self.previously_layer is not None:
            if self.previously_layer.type == TypeOfLayer.INPUT:
                delta = self.delta
                weight_delta = np.outer(delta, input_value)
            else:
                delta = self.delta
                values = self.previously_layer.values
                weight_delta = np.outer(delta, values)
            self.weight_delta = weight_delta

    def __change_weights(self, learning_rate: float):
        if self.type != TypeOfLayer.INPUT:
            weights = self.weights
            weights_delta = self.weight_delta
            self.weights = weights - self.multiply(learning_rate, weights_delta)
