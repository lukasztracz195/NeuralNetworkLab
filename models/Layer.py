from activations.Activation import Activation
import numpy as np

from enums.TypeOfLayer import TypeOfLayer


class Layer:
    def __init__(self, units: int,weight_min_value, weight_max_value, activation: Activation, type: TypeOfLayer):
        self.__units = units
        self.__activation = activation
        self.__values = np.zeros(self.__units)
        self.__delta = 0.0
        self.__weights = None
        self.__weight_delta = None
        self.__type = type
        self.__weight_min_value = weight_min_value
        self.__weight_max_value = weight_max_value
        self.__index = None

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