import numpy
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from models.classification import BaseClassifier


class TextCNNClassifier(BaseClassifier):
    """
    A simple TextCNN model used by Li et al. (in QRS'17).
    """

    def __init__(self, **kwargs):
        """
        Constructor of Keras TextCNN model.

        :key load_checkpoint: Determine whether the model loads the trained model from file.
            If it is True, parameter 'checkpoint_filename' is required. Otherwise, parameters
            of the network such as 'sample_length' and 'dict_size' will be required.
        :key checkpoint_filename:  The filename of the checkpoint.
        :key sample_length: Length of all samples.
        :key dict_size: Vocabulary size, i.e., the number of unique tokens.

        :key filter_num: [Optional] Number of the filters. Default is set to 10.
        :key hidden_num: [Optional] Number of nodes in fully-connection hidden layer.
            Default is set to 100.
        :key filter_length: [Optional] Length of the filter(s). Default is set to 5.
        :key embedding_size: [Optional] Length of embedding vector. Default is set to 30.
        :key retrain: [Optional]. If it is True, the 'fit' operation will retrain
            the model whether it is loaded from file.
        """
        super().__init__(**kwargs)
        if self.load_checkpoint is True:
            if self.checkpoint_filename is None:
                raise ValueError("Argument 'checkpoint_filename' is required since "
                                 "'load_checkpoint' is True.")
            self.model = load_model(self.checkpoint_filename)
        else:
            # Initialize a new CNN Network without training.
            # It should be trained using 'fit' method.
            self.sample_length = kwargs.get("sample_length", None)
            if self.sample_length is None:
                raise ValueError("Argument 'sample_length' must be "
                                 "specified, None got.")
            self.dict_size = kwargs.get("dict_size", None)
            if self.dict_size is None:
                raise ValueError("Argument 'dict_size' must be specified"
                                 ", None got.")
            # Parse optional parameters of the Network.
            self.filter_num = kwargs.get("filter_num", 10)
            self.hidden_num = kwargs.get("hidden_num", 100)
            self.filter_length = kwargs.get("filter_length", 5)
            self.embedding_size = kwargs.get("embedding_size", 30)
            self.epochs = kwargs.get("epochs", 15)
            self.batch_size = kwargs.get("batch_size", 32)
            # Build the keras CNN network.
            self.model = self.__build_keras_network()
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __build_keras_network(self):
        # Build Keras CNN Model.
        input = Input(shape=(self.sample_length,), dtype='float64')
        embedding = Embedding(self.dict_size + 1,
                              self.embedding_size,
                              input_length=self.sample_length,
                              trainable=True)(input)
        conv = Conv1D(self.filter_num,
                      self.filter_length,
                      padding='same',
                      strides=1,
                      activation='relu')(embedding)
        pooling = MaxPooling1D(pool_size=self.sample_length - self.filter_length + 1)(conv)
        hidden = Dense(self.hidden_num, activation="relu")(pooling)
        output = Dense(1, activation='sigmoid')(hidden)
        return Model(inputs=input, outputs=output)

    def fit(self, X, y):
        """
        Train the network use given data and labels.

        :param X: The train data.
        :param y: The labels for train data X.
        :return: The trained TextCNN classifier.
        """
        if not self.load_checkpoint:
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict_proba(self, X):
        """
        Predict the probabilities of each class for given test data X.

        :param X: The data to be predicted.
        :return: Probabilities of each class (label) for all test data.
        """
        raw_proba = self.model.predict(X)
        output_proba = []
        for proba in raw_proba:
            output_proba.append([1 - proba[0][0], proba[0][0]])
        return numpy.array(output_proba)

    def predict(self, X):
        """
        Predict the labels for given test data X.

        :param X: The data to be predicted.
        :return: Integer labels for each class for all test data.
        """
        proba_array = self.predict_proba(X)
        result = []
        for proba in proba_array:
            result.append(1 if proba[1] > proba[0] else 0)
        return numpy.array(result)

    def save(self, filename):
        self.model.save(filename)
