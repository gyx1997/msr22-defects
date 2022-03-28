"""
An simple LSTM model for source code-based defect prediction.
"""

from models import BaseModel
from models.classification import LSTMClassifier
from models.vectorization import SequenceVectorizer


class LSTMModel(BaseModel):
    """
    Model with semantic features extracted by Bi-directional LSTM.
    """

    def __init__(self, **kwargs):
        """
        Constructor of LSTMClassifier.
        The structure of this LSTM Classifier is based on LSTM model from traditional text classification tasks.
        It only contains an embedding layer, an Bi-directional LSTM layer and an output layer.

        :key global_dictionary: The global dictionary.
        :key checkpoint: The checkpoint of the model.

        :key handle_imbalance: [Optional] The model will duplicate the instances of minor class by
            oversampling strategy if it is set True. Default False.

        :key epochs: [Optional] The epochs for training the LSTM network. Default 15.
        :key dropout: [Optional] The dropout for LSTM. Default 0.05
        :key batch_size: [Optional] The batch_size for training the LSTM network. Too large batch_size may
            cause failure for convergence. Default 16.
        :key lstm_units: [Optional] The number of LSTM units in Bi-directional LSTM layer. Default 128.
        :

        """

        # Parent constructor.
        super().__init__(**kwargs)
        self._set_model_name("lstm")
        # LSTM model uses code token sequence as input. Thus, we initialize SequenceVectorizer here.
        self.vectorizer = SequenceVectorizer(self.global_dictionary, normalization=False)
        # Parse hyper-parameters.
        self.embedding_size = kwargs.get("embedding_size", 30)
        self.lstm_units = kwargs.get("lstm_units", 128)
        self.batch_size = kwargs.get("batch_size", 16)
        self.dropout = kwargs.get("dropout", 0.05)
        self.epochs = kwargs.get("epochs", 15)
        self.load_checkpoint = kwargs.get("load_checkpoint", False)

        self.model_settings = {"lstm_units": self.lstm_units,
                               "embedding_size": self.embedding_size,
                               "batch_size": self.batch_size,
                               "dropout": self.dropout,
                               "epochs": self.epochs}
        # Load pretrained model if necessary.
        if self.load_checkpoint is True:
            if self.checkpoint_filename is None:
                raise ValueError("Parameter 'checkpoint' is required if 'load_checkpoint' is True")

            print(self.get_model_checkpoint_name())

            self.classifier = LSTMClassifier(**self.model_settings,
                                             load_checkpoint=True,
                                             checkpoint_filename=self.__lstm_checkpoint())
        else:
            self.classifier = None

    def fit(self, X, y):
        if self.handle_imbalance is True:
            X, y = self.resampler.fit_resample(X, y)
        X = self.vectorizer.vectorize(X)
        # Build LSTM Network
        self.model_settings["train_sample_num"] = X.shape[0]
        self.model_settings["sample_length"] = X.shape[1]
        self.model_settings["dict_size"] = self.global_dictionary.token_count
        self.classifier = LSTMClassifier(**self.model_settings,
                                         checkpoint_filename=self.get_model_checkpoint_name())
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        super().predict(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict_proba(X)

    def get_checkpoint_param_string(self):
        return "%dlstm_units" % (self.model_settings["lstm_units"])

    def __lstm_checkpoint(self):
        return self.get_model_checkpoint_name() + "/" + "lstm"

    def save(self):
        self._remake_checkpoint_dir()
        self.classifier.save(self.__lstm_checkpoint())
