import sys
from typing import List, Dict

import numpy as np

from enums.TypeOfLayer import TypeOfLayer
from loss.Loss import Loss
from metrics.Metric import Metric
from models.Layer import Layer
from models.NeuralStatistics import NeuralStatistic


class NeuralNetwork:
    def __init__(self):
        self.__layers = dict()
        self.__x = None
        self.__predict = None
        self.__keys_forward: List[int] = list()
        self.__keys_backward: List[int] = list()
        self.__list_metrics_functions: List[Metric] = list()
        self.__loss_function = None
        self.__statistics: NeuralStatistic = NeuralStatistic()
        self.__used_add_layer = set()

    def add_layer(self, layer: Layer):
        index_for_layer = len(self.__layers)
        layer.index = index_for_layer
        self.__layers[index_for_layer] = layer
        if index_for_layer > 0:
            self.__keys_forward.append(index_for_layer)

    def add_weights(self, key: int, weights):
        layer = self.__layers[key]
        layer.weights = weights
        layer.delta = 0.0
        self.__used_add_layer.add(key)

    def compile(self, loss: Loss, metrics: List[Metric]):
        keys_for_backward = self.__keys_forward.copy()
        keys_for_backward.reverse()
        self.__keys_backward = keys_for_backward
        self.__loss_function = loss
        self.__list_metrics_functions = metrics
        for index_of_layer in self.__keys_forward:
            index_layer_before = index_of_layer - 1
            current_layer = self.__layers[index_of_layer]
            before_layer = self.__layers[index_layer_before]
            rows = current_layer.units
            columns = before_layer.units
            if index_of_layer not in self.__used_add_layer:
                current_layer.weights = np.random.uniform(low=current_layer.weight_min_value,
                                                          high=current_layer.weight_max_value,
                                                          size=(rows, columns))
                current_layer.delta = 0.0

    @property
    def x(self) -> np.ndarray:
        return self.__x

    @property
    def statistics(self):
        return self.__statistics

    @property
    def predict(self):
        return self.__predict

    def valid(self, x_test, y_test, debug=False):
        self.__teach(x_test, y_test, epochs=1, learning_rate=0.0, debug=debug)

    def fit(self, x, y, epochs: int, learning_rate: float, debug=False):
        self.__teach(x, y, epochs=epochs, learning_rate=learning_rate, debug=debug)

    def __teach(self, x, y, epochs: int, learning_rate: float, debug=False):
        self.__clear_statistics()
        units = self.__layers[0].units
        assert x.shape[1] == units, 'Input shape is different as number_neurons input layer'
        self.__x = x
        number_of_cycle = 0
        number_of_series = y.shape[0]
        self.__predict = np.zeros(y.shape)
        full_progress_bar = epochs * number_of_series
        sum_series = 0
        while True:
            number_of_current_series = 0
            while True:
                expected_value = y[number_of_current_series]
                input_value = self.__x[number_of_current_series]
                self.__forward(input_value=input_value)
                self.__backward(expected_value=expected_value, learning_rate=learning_rate, input_value=input_value)
                part_y_pred = self.__extract_last_y_pred()
                if debug:
                    message = 'part pred_y after epocs %d, series %d : %s' % (
                        number_of_cycle + 1, number_of_current_series + 1, part_y_pred)
                    print(message)
                self.__predict[number_of_current_series] = part_y_pred
                number_of_current_series += 1
                sum_series += 1
                self.__save_error_by_series(expected_value, part_y_pred)
                current_progress = sum_series
                self.__progress_bar(current=current_progress, total=full_progress_bar)
                if number_of_current_series == number_of_series:
                    self.__save_metrics_history(y, self.__predict)
                    self.__save_error_by_epocs(y, self.__predict)
                    self.__count_condensed_error()
                    break
            number_of_cycle += 1
            if number_of_cycle == epochs:
                break

    def __forward(self, input_value):
        served_input_layer = False
        last_layer = None
        for key in self.__keys_forward:
            layer = self.__layers[key]
            if served_input_layer:
                values = last_layer.values
                weights = layer.weights.T
                layer.values = self.__multiply(values, weights)
                if layer.type == TypeOfLayer.HIDDEN:
                    layer.values = layer.activation.count_value(layer.values)
                last_layer = layer
            else:
                weights = layer.weights.T
                layer.values = self.__multiply(input_value, weights)
                if layer.type == TypeOfLayer.HIDDEN:
                    if layer.activation is not None:
                        layer.values = layer.activation.count_value(layer.values)
                last_layer = layer
                served_input_layer = True

    def __backward(self, expected_value: float, learning_rate: float, input_value):
        self.__count_deltas(expected_value=expected_value)
        self.__count_weight_deltas(input_value=input_value)
        self.__change_weights(learning_rate=learning_rate)

    def __count_deltas(self, expected_value):
        served_output_layer = False
        last_layer = None
        for key in self.__keys_backward:
            layer = self.__layers[key]
            if served_output_layer:
                delta = last_layer.delta
                weights = last_layer.weights
                layer.delta = self.__multiply(delta, weights)
                if layer.activation is not None:
                    derivative_values = layer.activation.count_value_of_derivative(layer.values)
                else:
                    derivative_values = 1.0
                layer.delta = layer.delta * derivative_values
                last_layer = layer
            else:
                values = layer.values
                layer.delta = values - expected_value
                served_output_layer = True
                last_layer = layer

    def __count_weight_deltas(self, input_value):
        served_input_layer = False
        last_layer = None
        for key in self.__keys_forward:
            layer = self.__layers[key]
            if served_input_layer:
                delta = layer.delta
                values = last_layer.values
                layer.weight_delta = np.outer(delta, values)
                last_layer = layer
            else:
                delta = layer.delta
                weight_delta = np.outer(delta, input_value)
                layer.weight_delta = weight_delta
                last_layer = layer
                served_input_layer = True

    def __change_weights(self, learning_rate: float):
        for key in self.__keys_backward:
            layer = self.__layers[key]
            new_weight = layer.weights - learning_rate * layer.weight_delta
            layer.weights = new_weight

    def __multiply(self, a, b):
        if type(a) == np.ndarray and type(b) == np.ndarray:
            try:
                return a.dot(b)
            except:
                return a.dot(b.T)
        return a * b

    def __extract_last_y_pred(self):
        last_index_of_layer = len(self.__layers) - 1
        output_layer = self.__layers[last_index_of_layer]
        return output_layer.values

    def __save_metrics_history(self, y_true, y_pred):
        for metric in self.__list_metrics_functions:
            metric_value = metric.count_value(y_true, y_pred)
            dict_metrics = self.__statistics.history_metrics
            if 'accuracy' in dict_metrics:
                dict_metrics['accuracy'].append(metric_value)
            else:
                dict_metrics['accuracy'] = list()

    def __save_error_by_epocs(self, y_true, y_predict):
        error = self.__count_error(y_true, y_predict)
        self.__statistics.error_for_epocs.append(error)

    def __save_error_by_series(self, y_true, y_predict):
        error = self.__count_error(y_true, y_predict)
        self.__statistics.error_for_series.append(error)

    def __count_error(self, y_true, y_predict):
        if self.__loss_function is not None:
            return self.__loss_function.count_value(y_true, y_predict)

    def __count_condensed_error(self):
        self.__statistics.condensed_error.append(np.mean(self.__statistics.error_for_epocs))

    def __clear_statistics(self):
        self.__statistics.clear()

    def __progress_bar(self, current, total, barLength=20):
        percent = float(current) * 100 / total
        arrow = '-' * int(percent / 100 * barLength - 1) + '>'
        spaces = ' ' * (barLength - len(arrow))
        if current % 10 == 0:
            sys.stdout.write('\rTeaching progress: [%s%s] %d %%' % (arrow, spaces, percent))

