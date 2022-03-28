from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import LSTM, Dense, Input, CuDNNLSTM
from tensorflow.python.keras.models import load_model
import numpy
import copy
from models.classification import BaseClassifier


class LSTMClassifier(BaseClassifier):
    """
    The sequence classifier based on Bidirectional LSTM network.
    """

    def __init__(self, **kwargs):
        """
        Constructor of LSTM Network, which is based on Keras LSTM model.

        :key load_checkpoint: Optional. Determine whether the model loads the
            trained model from file. If it is True, parameter 'checkpoint_filename'
            is required. Otherwise, parameters of the network such as
            'sample_length' and 'dict_size' will be required. Default is False.

        :key checkpoint_filename: Required if 'load_checkpoint' is True. The
            filename of the checkpoint.
        :key retrain: Optional. If it is True, the 'fit' operation will retrain
            the model whether it is loaded from file.

        :key sample_length: Required if 'load_checkpoint' is False. Length of all samples.
        :key dict_size: Required. Vocabulary size, i.e., the number of unique tokens.

        :key filter_length: Optional. Length of the filter(s). Default is set to 5.
        :key embedding_size: Optional. Length of embedding vector. Default is set to 30.
        """
        # Parse required parameters.
        super().__init__(**kwargs)
        if self.load_checkpoint is True:
            if self.checkpoint_filename is None:
                raise ValueError("Argument 'checkpoint_filename' is required since 'load_checkpoint' is True.")
            self.retrain = kwargs.get("retrain", False)
            self.model = load_model(self.checkpoint_filename)
        else:
            self.sample_length = kwargs.get("sample_length", None)
            if self.sample_length is None:
                raise ValueError("Argument 'sample_length' must be specified, None got.")
            self.dict_size = kwargs.get("dict_size", None)
            if self.dict_size is None:
                raise ValueError("Argument 'dict_size' must be specified, None got.")
            # Parse hyper parameters of the Network.
            self.lstm_units = kwargs.get("lstm_units", 128)
            self.embedding_size = kwargs.get("embedding_size", 30)
            self.epochs = kwargs.get("epochs", 15)
            self.batch_size = kwargs.get("batch_size", 28)
            self.dropout = kwargs.get("dropout", 0.05)
            # Build the keras CNN network.
            self.model = self.__build_keras_network()
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __build_keras_network(self):
        # Build Keras LSTM Model.
        model = Sequential()
        model.add(Input(shape=(self.sample_length,),
                        dtype='float64'))
        model.add(Embedding(self.dict_size + 1,
                            self.embedding_size,
                            input_length=self.sample_length,
                            trainable=True))
        model.add(Bidirectional(LSTM(self.lstm_units,
                                     recurrent_activation="sigmoid",
                                     dropout=self.dropout)))
        model.add(Dense(1, activation='tanh'))
        return model

    def fit(self, X, y):
        """
        Train the network use given data and labels.

        :param X: The train data.
        :param y: The labels for train data X.
        :return: The trained LSTM Model.
        """

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict_proba(self, X):
        raw_proba = self.model.predict(X)
        output_proba = []
        for proba in raw_proba:
            positive_proba = (proba[0] + 1) / 2
            output_proba.append([1 - positive_proba, positive_proba])
        return numpy.array(output_proba)

    def predict(self, X):
        """
        Predict the given data X.

        :param X: The data to be predicted.
        :return: Integer labels for each class.
        """
        proba_array = self.predict_proba(X)
        result = []
        for proba in proba_array:
            result.append(1 if proba[1] > proba[0] else 0)
        return numpy.array(result)

    def save(self, filename):
        self.model.save(filename)
