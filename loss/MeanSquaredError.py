from loss.Loss import Loss


class MeanSquaredError(Loss):

    def count_value(self, y_true, y_pred):
        return ((y_true - y_pred)**2).mean(axis=None)