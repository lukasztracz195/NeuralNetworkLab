from loss.Loss import Loss

import numpy as np

class MeanSquaredError(Loss):

    def count_value(self, y_true, y_pred):
        return (np.subtract(y_true,y_pred)**2).mean(axis=None)