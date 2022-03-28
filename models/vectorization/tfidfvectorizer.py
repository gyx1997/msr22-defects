import numpy

from models.vectorization import BaseVectorizer


class TFIDFVectorizer(BaseVectorizer):
    """
    Vectorizer for transforming a code token sequence into a vector of token frequency-based features.
    """
    def __init__(self, global_dictionary, **kwargs):
        """
        Constructor of SequenceVectorizer.

        :param global_dictionary: The global dictionary.
        :key normalization: [Optional] The flag for normalizing the identifiers. Default is False.
        :key use_idf: [Optional] The flag for whether uses inverse document frequency (IDF). Default is False.
        """
        super(TFIDFVectorizer, self).__init__(global_dictionary, **kwargs)
        self.use_idf = kwargs.get("use_idf", False)
        # If inverse document frequency is used, the vector will also be normalized.
        if self.use_idf is True:
            self.normalization = True

    def vectorize(self, X):
        """
        Vectorize given token sequences.

        :param X: The given token sequences.
        :return: A 2d-numpy array of vectorized token sequences.
        """
        num_sequences = len(X)

        x_matrix = numpy.zeros((num_sequences, self.global_dictionary.token_count))
        num_tokens_of_sequences = []
        for i in range(0, num_sequences):
            x = X[i]
            num_tokens_of_sequences.append(len(x))
            for j in range(0, len(x)):
                token_id = self.global_dictionary.get_token_id(x[j])
                if token_id >= 0:
                    if self.use_idf is True:
                        # tfidf = tf * idf, where tf = sum(I(x)) / len(c), and I(x) is whether token x is in sequence.
                        x_matrix[i, token_id] += (1 * self.global_dictionary.inverse_document_frequency[self.global_dictionary.id2token[token_id]])
                    else:
                        x_matrix[i, token_id] += 1

        # Normalization is necessary when using TF-IDF model.
        # It can also be conducted on TF model.
        if self.normalization is True:
            # We do normalization first.
            for i in range(0, num_sequences):
                x_matrix[i, :] /= (num_tokens_of_sequences[i] + 1)
            pass

        return x_matrix
