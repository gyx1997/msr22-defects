"""
Models for Wang et al. (in ICSE'16)
"""
import pickle
from sklearn.linear_model import LogisticRegression
from models.vectorization import SequenceVectorizer
from models import BaseModel
from dbn.models import UnsupervisedDBN


class DBNModel(BaseModel):
    """
    Model with semantic features extracted by Deep belief network.
    """

    def __init__(self, **kwargs):
        """
        Constructor of DBNModel. This simple DBN model assumes the number of
        nodes in all layer(s) are the same according to Wang et al. Though, you can modify this model
        by specify the nodes for each layer manually.

        :key load_checkpoint: The model will be loaded from trained checkpoint (file) instead of training
            a new model if it is set True. Note that checkpoint_filename will be required if load_checkpoint
            is set True.
        :key checkpoint_filename: The filename of trained checkpoint. It is required if load_checkpoint is
            set True.


        :key hidden_layers: [Optional] Number of hidden layers in Deep Belief Network. Default 4.
        :key nodes: Optional. Number of nodes in each hidden layer. Default 256.
        :key num_iteration: [Optional] The number of iterations for Deep Belief Network. Default 200.
        :key classifier: [Optional] The classifier object for binary classification. It must have method
            'fit' and 'predict'. Default is LogisticRegression(C=100, max_iter=10000, solver="liblinear").
        :key handle_imbalance: [Optional] The model will use imblearn.SMOTE, an over-sampling technique to re-balance
            the positive and negative instances if it is set True. Default False.
        """
        super().__init__(**kwargs)
        self._set_model_name("dbn")
        # DBN model uses code token sequence as input. Thus, we initialize SequenceVectorizer with normalization here.
        self.vectorizer = SequenceVectorizer(self.global_dictionary, normalization=True)

        self.hidden_layers = kwargs.get("hidden_layers", 4)
        self.nodes_per_layer = kwargs.get("nodes", 256)
        self.num_iteration = kwargs.get("iteration", 200)

        if self.load_checkpoint is True:
            # Load trained model from file
            # Unsupervised dbn feature extractor.
            fd = open(self.__unsupervised_dbn_checkpoint(), "rb+")
            self.unsupervised_dbn = pickle.load(fd)
            fd.close()
            # Classifier.
            fd = open(self.__classifier_checkpoint(), "rb+")
            self.classifier = pickle.load(fd)
            fd.close()
        else:
            # Construct unsupervised deep belief network.
            self.unsupervised_dbn = UnsupervisedDBN(
                hidden_layers_structure=[self.nodes_per_layer for _ in range(0, self.hidden_layers)],
                n_epochs_rbm=self.num_iteration
            )
            self.classifier = kwargs.get("classifier", LogisticRegression(C=100, max_iter=10000, solver="liblinear"))

    def fit(self, X, y):
        if self.handle_imbalance is True:
            X, y = self.resampler.fit_resample(X, y)
        X = self.vectorizer.vectorize(X)
        self.unsupervised_dbn.fit(X)
        X_transformed = self.unsupervised_dbn.transform(X)
        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X = self.vectorizer.vectorize(X)
        X_transformed = self.unsupervised_dbn.transform(X)
        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        X = self.vectorizer.vectorize(X)
        X_transformed = self.unsupervised_dbn.transform(X)
        return self.classifier.predict_proba(X_transformed)

    def get_checkpoint_param_string(self):
        return "%dlayers-%dnodes-%diterations" % (self.hidden_layers,
                                                  self.nodes_per_layer,
                                                  self.num_iteration)
        pass

    def save(self):
        self._remake_checkpoint_dir()
        fd = open(self.__unsupervised_dbn_checkpoint(), "wb+")
        pickle.dump(self.unsupervised_dbn, fd)
        fd.close()

        fd = open(self.__classifier_checkpoint(), "wb+")
        pickle.dump(self.classifier, fd)
        fd.close()

    def __unsupervised_dbn_checkpoint(self):
        return self.get_model_checkpoint_name() + "/" + "unsupervised_dbn"

    def __classifier_checkpoint(self):
        return self.get_model_checkpoint_name() + "/" + "classifier"

