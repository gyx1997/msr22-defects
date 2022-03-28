from abc import ABC

from sklearn.linear_model import LogisticRegression
import numpy
import copy
import models.basemodel
import pickle

from models.vectorization.tfidfvectorizer import TFIDFVectorizer


class TFIDFModel(models.basemodel.BaseModel):
    """
    Model based on term frequency (TF) and term frequency-inverse document frequency (TF-IDF).
    """

    def __init__(self, **kwargs):
        """
        Constructor of TFIDFModel
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.classifier = kwargs.get("classifier", LogisticRegression())
        self.use_idf = kwargs.get("use_idf", False)
        if self.use_idf is True:
            # TF-IDF model. In this situation, the normalization of token frequency is required.
            self._set_model_name("tfidf")
            self.vectorizer = TFIDFVectorizer(self.global_dictionary, use_idf=True, normalization=True)
            self.normalization = True
        else:
            # TF model. In this situation, the normalization can be specified manually.
            self._set_model_name("tf")
            self.normalization = kwargs.get("normalization", False)
            self.vectorizer = TFIDFVectorizer(self.global_dictionary, use_idf=False, normalization=self.normalization)
        if self.load_checkpoint:
            fd = open(self.__tfidf_checkpoint(), "rb+")
            self.classifier = pickle.load(fd)
            fd.close()

    def fit(self, X, y):
        super(TFIDFModel, self).fit(X, y)
        if self.handle_imbalance is True:
            X, y = self.resampler.fit_resample(X, y)
        X = self.vectorizer.vectorize(X)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        super(TFIDFModel, self).predict(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        super(TFIDFModel, self).predict_proba(X)
        X = self.vectorizer.vectorize(X)
        return self.classifier.predict_proba(X)

    def get_checkpoint_param_string(self):
        return "normalized" if self.normalization is True else "raw"

    def save(self):
        self._remake_checkpoint_dir()
        fd = open(self.__tfidf_checkpoint(), "wb+")
        pickle.dump(self.classifier, fd)
        fd.close()

    def __tfidf_checkpoint(self):
        return self.get_model_checkpoint_name() + "/classifier"
