import copy
from abc import ABCMeta, abstractmethod
import numpy
import models.vectorization
import models.resampling
import models.classification
from preprocess import GlobalDictionary
from utils.io import rmdir
from os import makedirs


class BaseModel(metaclass=ABCMeta):
    """
    Base class of source code-based defect prediction models.
    """

    def __init__(self, **kwargs):
        self.model_name = ""

        self.checkpoint_filename = kwargs.get("checkpoint", None)
        self.load_checkpoint = kwargs.get("load_checkpoint", False)
        self.handle_imbalance = kwargs.get("handle_imbalance", False)
        self.global_dictionary = kwargs.get("global_dictionary", None)  # type: GlobalDictionary
        self.classifier = kwargs.get("classifier", None)  # type: models.classification.BaseClassifier
        self.vectorizer = kwargs.get("vectorizer", None)  # type: models.vectorization.BaseVectorizer
        self.resampler = kwargs.get("resampler", None)  # type: models.resampling.BaseResampler
        if self.resampler is None:
            self.handle_imbalance = False
        pass

    def fit(self, X, y):
        """
        Train the model.

        :param X: Required. List of samples.
        :param y: Required. Labels.

        :rtype: None
        """
        self.__check_arguments()

    def predict(self, X):
        """
        Make the prediction by the trained model.

        :param X: Required. List of samples.

        :return: Returns a numpy 1d-array which contains the predicted labels.
        """
        self.__check_arguments()
        self.__check_classifier()

    def predict_proba(self, X):
        """
        Make the predictions by the trained model, and returns the probabilities for each class.
        Note that in defect prediction tasks they are all binary classification.

        :param X: Required. List of samples.

        :return: Returns a numpy 2d-array which elements are numpy 1d-array
                contains probabilities for each class.
        """
        self.__check_arguments()
        self.__check_classifier()

    @abstractmethod
    def get_checkpoint_param_string(self):
        """
        Get the string of model parameters.
        Only used for build filename of model checkpoint.

        :return: A string of model parameters.
        :rtype: str
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the model to checkpoint file.

        :rtype: None
        """
        pass

    def _remake_checkpoint_dir(self):
        """
        Remove the old checkpoint directory and make a new one.
        """
        rmdir(self.get_model_checkpoint_name())
        makedirs(self.get_model_checkpoint_name())

    def get_model_checkpoint_name(self):
        """
        Get the full file name of the model under current parameters.

        :return: A string of model checkpoint filename.
        :rtype: str
        """
        return self.checkpoint_filename \
               + "-" + self.get_checkpoint_param_string() \
               + "." + self.model_name \
               + ".model"
        pass

    def __check_arguments(self):
        if self.global_dictionary is None:
            raise ValueError("Argument 'global_dictionary' must be specified.")
        if self.vectorizer is None:
            raise ValueError("Argument 'vectorizer' must be specified.")

    def __check_classifier(self):
        if self.classifier is None:
            raise RuntimeError("Classifier cannot be used without training.")

    def _set_model_name(self, model_type):
        self.model_name = model_type
