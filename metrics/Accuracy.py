from metrics.Metric import Metric


class Accuracy(Metric):

    def count_value(self, y_true, y_pred):
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / len(y_true) * 100
