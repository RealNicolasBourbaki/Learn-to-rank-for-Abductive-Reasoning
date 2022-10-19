__author__ = '{Esra Dönmez}'


class Accuracy:
    """
    A class representing simple accuracy metric.
    """
    def __init__(self):
        pass

    def binary_accuracy(self, gold_labels, predicted_labels):
        """
        Args:
          gold_labels: ground truth labels
          predicted_labels: predictions from the model
        
        Returns:
          int: accuracy
        """
        correct = 0
        for i in range(len(gold_labels)):
            if gold_labels[i] == predicted_labels[i]:
                correct += 1
        return correct / float(len(gold_labels)) * 100.0
