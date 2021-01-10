from metrics.Metric import Metric
import numpy as np


class MeanAccuracy(Metric):

    def count_value(self, true_labels, classification_scores):
        assert type(true_labels) == type(classification_scores), "Types of predicted and true labels is not the same"
        assert true_labels.shape == classification_scores.shape, "Size of predicted and true labels is not the same"
        value = np.mean(np.argmax(classification_scores, axis=1) == true_labels)
        return value
