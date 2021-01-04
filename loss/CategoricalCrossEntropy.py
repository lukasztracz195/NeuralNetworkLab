from loss.Loss import Loss
import numpy as np


class CategoricalCrossEntropy(Loss):

    def count_value(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 0.0000001, 1 - 0.0000001)
        return -np.sum(y_true * np.log(y_pred))
