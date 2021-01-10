import sys
from typing import List, Dict

import numpy as np

from enums.TypeOfData import TypeOfData
from enums.TypeOfLayer import TypeOfLayer
from loss.Loss import Loss
from metrics.Metric import Metric
from models.Layer import Layer
from models.NeuralStatistics import NeuralStatistic
import time


class NeuralNetwork:
    def __init__(self):
        self.__layers = dict()
        self.__x = None
        self.__predict = dict()
        self.__test_predict = None
        self.__list_metrics_functions: List[Metric] = list()
        self.__loss_function = None
        self.__statistics: NeuralStatistic = NeuralStatistic()
        self.__used_add_layer = set()
        self.__logs = list()
        self.__output_layer = None
        self.__input_layer = None
        self.__previously_layer = None

    def add_layer(self, layer: Layer):

        if self.__input_layer is None or layer.type == TypeOfLayer.INPUT:
            self.__input_layer = layer
        if self.__output_layer is None and layer.type == TypeOfLayer.OUTPUT:
            self.__output_layer = layer
        if self.__previously_layer is not None:
            self.__previously_layer.next_layer = layer
            layer.previously_layer = self.__previously_layer
        self.__previously_layer = layer
        index_for_layer = len(self.__layers)
        layer.index = index_for_layer
        self.__layers[index_for_layer] = layer

    def add_weights(self, key: int, weights):
        layer = self.__layers[key]
        layer.weights = weights
        layer.delta = 0.0
        self.__used_add_layer.add(key)

    def compile(self, loss: Loss, metrics: List[Metric]):
        self.__loss_function = loss
        self.__list_metrics_functions = metrics
        current_layer = self.__input_layer
        while current_layer is not None:
            if current_layer.previously_layer is not None:
                previously_layer = current_layer.previously_layer
                rows = current_layer.units
                columns = previously_layer.units
                if current_layer.index not in self.__used_add_layer:
                    current_layer.weights = np.random.uniform(low=current_layer.weight_min_value,
                                                              high=current_layer.weight_max_value,
                                                              size=(rows, columns))
                current_layer.delta = 0.0
            current_layer = current_layer.next_layer

    @property
    def x(self) -> np.ndarray:
        return self.__x

    @property
    def test_predict(self) -> np.ndarray:
        return self.__test_predict

    @property
    def statistics(self):
        return self.__statistics

    @property
    def predict(self):
        return self.__predict

    def valid(self, x_test, y_test, type_data=TypeOfData.TEST, debug=False):
        self.__teach(x_test, y_test, epochs=1, learning_rate=0.0, type_data=type_data, debug=debug)

    def fit(self, x, y, epochs: int, learning_rate: float, type_data=TypeOfData.TRAINING, debug=False):

        start = time.time()
        self.__teach(x, y, epochs=epochs, learning_rate=learning_rate, type_data=type_data, debug=debug)
        end = time.time()
        print("\nTime teaching: ", end - start, " s")
        if debug:
            self.__print_logs()

    def __teach(self, x, y, epochs: int, learning_rate: float, type_data=TypeOfData.TRAINING, debug=False):
        self.__clear_statistics()
        units = self.__layers[0].units
        assert x.shape[1] == units, 'Input shape is different as number_neurons input layer'
        self.__x = x
        number_of_cycle = 0
        number_of_series = y.shape[0]
        self.__predict[type_data] = np.zeros(y.shape)
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
                message = 'part pred_y after epocs %d, series %d : %s' % (
                    number_of_cycle + 1, number_of_current_series + 1, part_y_pred)
                self.__logs.append(message)
                self.__predict[type_data][number_of_current_series] = part_y_pred
                number_of_current_series += 1
                sum_series += 1
                self.__save_error_by_series(expected_value, part_y_pred)
                current_progress = sum_series
                self.__progress_bar(current=current_progress, total=full_progress_bar)
                if number_of_current_series == number_of_series:
                    self.__save_metrics_history(y, self.__predict[type_data])
                    self.__save_error_by_epocs(y, self.__predict[type_data])
                    self.__count_condensed_error()
                    break
            number_of_cycle += 1
            if number_of_cycle == epochs:
                break

    def __forward(self, input_value):
        current_layer = self.__input_layer
        while current_layer is not None:
            current_layer.forward(input_value)
            current_layer = current_layer.next_layer

    def __backward(self, expected_value: float, learning_rate: float, input_value):
        current_layer = self.__output_layer
        while current_layer is not None:
            current_layer.backward(expected_value, input_value)
            current_layer = current_layer.previously_layer
        current_layer = self.__output_layer
        while current_layer is not None:
            current_layer.change_weights(learning_rate)
            current_layer = current_layer.previously_layer

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
            sys.stdout.write('\rTeaching progress: [%s%s] %f %%' % (arrow, spaces, percent))

    def __print_logs(self):
        print("\n")
        for log in self.__logs:
            print(log)
