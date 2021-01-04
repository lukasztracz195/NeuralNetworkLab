from loss.Loss import Loss
import numpy as np


class BinaryCrossEntropy(Loss):

    def count_value(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 0.0000001, 1 - 0.0000001)
        return - y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1 - y_pred)
