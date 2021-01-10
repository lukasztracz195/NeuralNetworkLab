from activations.Activation import Activation
import numpy as np

from enums.TypeOfLayer import TypeOfLayer


class Layer:
    def __init__(self, units: int, weight_min_value=0.0, weight_max_value=1.0, activation=None,
                 type=TypeOfLayer.UNDEFINED):
        self.__units = units
        self.__activation = activation
        self.__values = np.zeros(self.__units)
        self.__delta = 0.0
        self.__weights = None
        self.__weight_delta = None
        self.__type: TypeOfLayer = type
        self.__weight_min_value = weight_min_value
        self.__weight_max_value = weight_max_value
        self.__index = None
        self.__previously_layer = None
        self.__next_layer = None
        self.__logs = list()
        assert type != TypeOfLayer.UNDEFINED, 'Layer not have defined type'

    @property
    def units(self) -> int:
        return self.__units

    @property
    def weight_min_value(self) -> float:
        return self.__weight_min_value

    @property
    def weight_max_value(self) -> float:
        return self.__weight_max_value

    @property
    def activation(self) -> Activation:
        return self.__activation

    @property
    def values(self) -> np.ndarray:
        return self.__values

    @values.setter
    def values(self, value: np.ndarray):
        self.__values = value

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, value):
        self.__index = value

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, value: float):
        self.__delta = value

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = value

    @property
    def weight_delta(self):
        return self.__weight_delta

    @weight_delta.setter
    def weight_delta(self, value):
        self.__weight_delta = value

    @property
    def type(self) -> TypeOfLayer:
        return self.__type

    @property
    def previously_layer(self):
        return self.__previously_layer

    @previously_layer.setter
    def previously_layer(self, value):
        self.__previously_layer = value

    @property
    def next_layer(self):
        return self.__next_layer

    @next_layer.setter
    def next_layer(self, value):
        self.__next_layer = value

    def backward(self, expected_value: float, input_value):
        pass

    def change_weights(self, learning_rate: float):
        pass

    def multiply(self, a, b):
        if type(a) == np.ndarray and type(b) == np.ndarray:
            try:
                return a.dot(b)
            except:
                return a.dot(b.T)
        return a * b
