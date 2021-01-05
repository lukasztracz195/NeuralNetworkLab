from pygame.tests.draw_test import RED

from activations.ReLU import ReLU
from activations.Sigmoid import Sigmoid
from activations.SoftMax import SoftMax
from activations.Tanh import Tanh
from enums.TypeOfLayer import TypeOfLayer
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from loss.MeanSquaredError import MeanSquaredError
from metrics.Accuracy import Accuracy
from metrics.CustomAccuracy import CustomAccuracy
from models import NeuralStatistics
from models.Layer import Layer
from models.NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import os

training_file = "training_colors.txt"
test_file = "test_colors.txt"
global_path_to_training_file = os.path.abspath(training_file)
global_path_to_test_file = os.path.abspath(test_file)


def extract_set_data(path_to_file: str):
    f = open(path_to_file, "r")
    text = f.read()
    lines = text.split('\n')
    list_data = []
    for line in lines:
        list_data.append(np.array(line.split(' '), dtype=float))
    f.close()
    return np.asarray(list_data)


def prepare_expected_value(y, xy):
    tmp_array = np.zeros(xy.shape)
    index = 0
    for yt in y:
        index_color = int(yt[0] - 1)
        tmp_array[index, index_color] = 1
        index += 1
    return tmp_array


training_set = extract_set_data(global_path_to_training_file)
training_set_x = training_set[:, :3]
training_set_y = training_set[:, 3:4]
training_expected_value = prepare_expected_value(training_set_y, training_set)

test_set = extract_set_data(global_path_to_test_file)
test_set_x = test_set[:, :3]
test_set_y = test_set[:, 3:4]
test_expected_value = prepare_expected_value(test_set_y, test_set)

model = NeuralNetwork()

model.add_layer(Layer(units=3, type=TypeOfLayer.INPUT))  # 0
model.add_layer(Layer(units=5, activation=ReLU(), type=TypeOfLayer.HIDDEN))  # 1
model.add_layer(Layer(units=4, activation=SoftMax(), type=TypeOfLayer.OUTPUT))  # 2

model.compile(loss=MeanSquaredError(), metrics=[CustomAccuracy()])

model.fit(x=training_set_x, y=training_expected_value, epochs=10, learning_rate=0.000001, debug=False)

statistics: NeuralStatistics = model.statistics

history_error = statistics.error_for_epocs
history_accuracy = statistics.history_metrics['accuracy']
condensed_error = statistics.condensed_error
plt.figure(figsize=(20, 10))
plt.plot(history_error, '-*')
plt.xlabel('Number of epocs')
plt.ylabel('Value of loss function')
plt.show()

plt.plot(history_accuracy, '-+')
plt.xlabel('Number of epocs')
plt.ylabel('Value of accuracy')
plt.show()

plt.plot(condensed_error, '-o')
plt.xlabel('Number of epocs')
plt.ylabel('Value of condenced error')
plt.show()
print("error:", statistics.condensed_error[len(statistics.condensed_error)-1])

model.fit(x=test_set_x, y=test_expected_value, epochs=1, learning_rate=0, debug=False)
predict = model.predict


def prepare_predict(predict):
    predict_color_array = np.zeros(predict.shape[0])
    index = 0
    for record in predict:
        max_from_record = max(record)
        for index_with_max in range(predict.shape[1]):
            if predict[index, index_with_max] == max_from_record:
                predict_color_array[index] = index_with_max
                index += 1
                break
    return predict_color_array


predicted_values = prepare_predict(predict)
print('Predicted value')
print(predicted_values)
print('Test value')
test_values = test_set_y.flatten()
print(test_values)
mse = Accuracy()
accuracy = mse.count_value(test_values, predicted_values)
print('accuracy:', accuracy, ' %')
