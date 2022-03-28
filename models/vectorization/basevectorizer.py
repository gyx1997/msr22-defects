from abc import ABCMeta, abstractmethod
from preprocess.globaldictionary import GlobalDictionary


class BaseVectorizer(metaclass=ABCMeta):
    """
    Base class for all vectorizers.
    """
    def __init__(self, global_dictionary, **kwargs):
        """
        Constructor of SequenceVectorizer.

        :param global_dictionary: The global dictionary.
        :key normalization: [Optional] The flag for normalizing the identifiers. Default is False.
        """
        self.global_dictionary = global_dictionary  # type: GlobalDictionary
        if not isinstance(self.global_dictionary, GlobalDictionary):
            raise ValueError("Argument 'global_dictionary' must be an instance of 'preprocess.GlobalDictionary'.")
        self.normalization = kwargs.get("normalization", False)

    @abstractmethod
    def vectorize(self, X):
        """
        Vectorize given token sequences.

        :param X: The given token sequences.
        :return: A 2d-numpy array of vectorized token sequences.
        """
        pass