"""
Models for Li et al. (in QRS'17)
"""

import models.basemodel
from models.classification import TextCNNClassifier
from models.vectorization import SequenceVectorizer


class CNNModel(models.basemodel.BaseModel):
    """
    Model with semantic features extracted by TextCNN provided by Li et al. (in QRS'17).
    """

    def __init__(self, **kwargs):
        """
        Constructor of CNNModel.
        The structure of this CNN model is based on Li et al. (in QRS'17).

        :key global_dictionary: The global dictionary of code tokens.
        :key checkpoint: The checkpoint file of pretrained CNN network. If
        'load_pretrain' is set True, the network parameters will be loaded from the checkpoint file.
        The DBN network will be saved into this checkpoint file after the training process finished.

        :key handle_imbalance: [Optional] The model will duplicate the instances of minor class by
            oversampling strategy if it is set True. Default False.
        """

        super().__init__(**kwargs)
        self._set_model_name("cnn")
        # CNN model uses code token sequence as input. Thus, we initialize SequenceVectorizer here.
        self.vectorizer = SequenceVectorizer(self.global_dictionary, normalization=False)
        # Hyper parameters of CNN network.
        self.filter_num = kwargs.get("filter_num", 10)
        self.hidden_num = kwargs.get("hidden_num", 100)
        self.filter_length = kwargs.get("filter_length", 5)
        self.embedding_size = kwargs.get("embedding_size", 30)
        self.cnn_settings = {"hidden_num": self.hidden_num,
                             "filter_num": self.filter_num,
                             "filter_length": self.filter_length,
                             "embedding_size": self.embedding_size}
        if self.load_checkpoint is True:
            if self.checkpoint_filename is None:
                raise ValueError("Parameter 'checkpoint' is required if 'load_checkpoint' is True")
            self.classifier = TextCNNClassifier(load_checkpoint=True,
                                                checkpoint_filename=self.__cnn_checkpoint())
        else:
            self.classifier = None

    def fit(self, X, y):
        """
        Train the semantic CNN Model.

        :param X: List of sequences, where each sequence is a list of tokens.
        :param y: Vector of actual labels.
        :rtype: CNNModel
        """
        super().fit(X, y)
        # Handle class imbalance problem with specified resampling technique(s).
        if self.handle_imbalance is True:
            X, y = self.resampler.fit_resample(X, y)
        # Vectorize the training data.
        X = self.vectorizer.vectorize(X)
        # Build CNN Network
        self.cnn_settings["train_sample_num"] = X.shape[0]
        self.cnn_settings["sample_length"] = X.shape[1]
        self.cnn_settings["dict_size"] = self.global_dictionary.token_count
        self.classifier = TextCNNClassifier(**self.cnn_settings)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the labels of test data.

        :param X: List of sequences which is the test data.
        :return: A numpy array of predicted labels.
        """
        super().predict(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Predict the probabilities of each class for test data.

        :param X: List of sequences which is the test data.
        :return: A 2d-array of probabilites for each class (0 for clean, and 1 for defective).
        """
        super().predict(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict_proba(X)

    def get_checkpoint_param_string(self):
        return "%dfilters-%dwidth-%dhidden" % (self.cnn_settings["filter_num"],
                                               self.cnn_settings["filter_length"],
                                               self.cnn_settings["hidden_num"])

    def save(self):
        self._remake_checkpoint_dir()
        self.classifier.save(self.__cnn_checkpoint())

    def __cnn_checkpoint(self):
        return self.get_model_checkpoint_name() + "/" + "cnn"
