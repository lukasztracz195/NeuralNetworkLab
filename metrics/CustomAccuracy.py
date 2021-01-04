import numpy as np

from metrics.Metric import Metric


class CustomAccuracy(Metric):

    def count_value(self, y_true, y_pred):
        corrects = 0
        number_of_records = y_true.shape[0]
        y_pred_tmp = np.zeros_like(y_pred)
        y_pred_tmp[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
        index = 0
        for record in y_pred_tmp:
            if np.array_equal(record, y_true[index]):
                corrects += 1
            index += 1
        return (corrects / number_of_records) * 100
