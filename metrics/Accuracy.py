from metrics.Metric import Metric

import numpy as np


class Accuracy(Metric):

    def count_value(self, true_labels, classification_scores):
        assert type(true_labels) == type(classification_scores), "Types of predicted and true labels is not the same"
        assert true_labels.shape == classification_scores.shape, "Size of predicted and true labels is not the same"
        if type(true_labels) == np.ndarray:
            if (np.ndim(true_labels) == 1):
                return self.__accuracy_for_vector(true_labels, classification_scores)
            else:
                return self.__accuracy_for_matrix(true_labels, classification_scores)

    def __accuracy_for_vector(self, true_labels, classification_scores):
        if (true_labels == classification_scores).all():
            return 100.0
        return self.__square_magnitude([classification_scores[i] - true_labels[i] for i in range(len(true_labels))]) / len(classification_scores)

    def __square_magnitude(self, vector):
        return sum(x * x for x in vector)

    def __accuracy_for_matrix(self, true_labels, classification_scores):
        corrects = 0
        number_of_records = true_labels.shape[0]
        y_pred_tmp = np.zeros_like(classification_scores)
        y_pred_tmp[np.arange(len(classification_scores)), classification_scores.argmax(1)] = 1
        index = 0
        for record in y_pred_tmp:
            if np.array_equal(record, true_labels[index]):
                corrects += 1
            index += 1
        value = (corrects / number_of_records) * 100
        return value


