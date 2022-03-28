from abc import ABCMeta, abstractmethod

import explanation.feature_representation


class BaseExplanation(metaclass=ABCMeta):
    """
    Base class for explanation result.
    """

    def __init__(self, **kwargs):
        """
        Constructor of BaseExplanation.

        :key key_features: The most informative atomic features returned by the explanator.
            It should be arranged as a list of tuple, i.e.,
            [(feature_1, probability_of_feature_1), (feature_2, probability_of_feature_2), ...].
        """

        self._key_features = kwargs.get("key_features", None)

    @property
    def key_features(self):
        """
        Returns the key features. The key features should be arranged as a list of tuple, i.e.,
            [(feature_1, probability_of_feature_1), (feature_2, probability_of_feature_2), ...].
        """

        # Do type check first.
        if self._key_features is None:
            raise AttributeError("Instance of " + self.__class__.__name__ +
                                 " does not have attribute 'key_features'. ")

        return self._key_features


class BaseExplanator(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        """
        Constructor of the explanation method.
        :key classifier: The instance of the classifier.
            Must have 'predict' function.
        """

        self._classifier = kwargs.get("classifier", None)
        if self._classifier is None:
            raise ValueError("The black-box classifier must be specified.")

        if not callable(self._classifier.predict):
            raise AttributeError("The black-box classifier does not implement 'predict' method.")

        self.can_use_predict_proba = callable(self._classifier.predict_proba)

    @abstractmethod
    def explain(self, X, **kwargs):
        """
        Explain the given instance X.
        :param X: The instance to be explained.
        :return: The explanation result.
        """
        pass

    pass

    @property
    def classifier(self):
        """
        Returns the black-box classifier assigned with this explanator.
        """
        return self._classifier

    def _predict(self, X):
        """

        :param X:
        :return:
        """
        if isinstance(X, explanation.feature_representation.Instance):
            raise TypeError("Only type of 'explanation.feature_representation.Instance' "
                            "could be used in BaseExplanationMethod._predict")

        return self.classifier.predict([self.__transform(X)])

    def _predict_proba(self, X):
        """

        :param X:
        :return:
        """
        if not self.can_use_predict_proba:
            raise RuntimeError("The black-box classifier does not support calling 'predict_proba'.")

        if isinstance(X, explanation.feature_representation.Instance):
            raise TypeError("Only type of 'explanation.feature_representation.Instance' "
                            "could be used in BaseExplanationMethod._predict")

        return self.classifier.predict_proba([self.__transform(X)])

    def __transform(self, X):
        """
        Transform list of atomic features into list of tokens, which could be fed
            into the classifier directly.

        :param X: List of atomic features.
        :return: List of tokens.
        """
        return explanation.feature_representation.features_to_tokens(X)
