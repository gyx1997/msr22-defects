from abc import ABCMeta, abstractmethod


class BaseClassifier(metaclass=ABCMeta):
    """
    Base class of classifiers for source code-based defect prediction models.
    """

    def __init__(self, **kwargs):
        self.load_checkpoint = kwargs.get("load_checkpoint", False)
        self.checkpoint_filename = kwargs.get("checkpoint_filename", None)
        self.verbose = kwargs.get("verbose", True)
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Train the network use given data and labels.

        :param X: The train data.
        :param y: The labels for train data X.
        :return: The trained TextCNN classifier.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict the probabilities of each class for given test data X.

        :param X: The data to be predicted.
        :return: Probabilities of each class (label) for all test data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the labels for given test data X.

        :param X: The data to be predicted.
        :return: Integer labels for each class for all test data.
        """
        pass

    @abstractmethod
    def save(self, filename):
        pass

