import copy
import numpy
import models.vectorization
from models.vectorization import BaseVectorizer

class SequenceVectorizer(BaseVectorizer):
    """
    Vectorizer for transforming a code token sequence into a vector of unique identifiers (of code tokens).
    """
    def __init__(self, global_dictionary, **kwargs):
        super(SequenceVectorizer, self).__init__(global_dictionary, **kwargs)

    def vectorize(self, X):
        normalization = self.normalization
        # Build the sequence vector.
        normalized_matrix = []
        for sequence in X:
            # Define the transformed vector for current sequence.
            transformed_vector = []
            # Check all token in the sequence and append the token which satisfies the minimum
            # appearance criterion into the transformed vector.
            for token_element in sequence:
                # If the token satisfies the criterion, append it to the vector.
                if self.global_dictionary.token2id.__contains__(token_element):
                    transformed_vector.append(self.global_dictionary.token2id[token_element])
            normalized_matrix.append(transformed_vector)
        # For sequence of different length, we should do padding operation here.
        for transformed_sequence in normalized_matrix:
            for _ in range(len(transformed_sequence), self.global_dictionary.max_sequence_length):
                transformed_sequence.append(0)
        # Convert the 2d python list into numpy matrix (2darray).
        npmat_final_matrix = numpy.array(normalized_matrix)
        # Normalize the matrix if needed.
        if normalization is True:
            def min_max_norm(x, _min, _max):
                # Get the vector of min and max.
                min_value = numpy.array([_min for _ in range(x.shape[0])])
                max_value = numpy.array([_max for _ in range(x.shape[0])])
                # Use vector operation to do normalization.
                return (x - min_value) / (max_value - min_value)
            normalized_matrix = []
            for i in range(0, npmat_final_matrix.shape[0]):
                # Get the i-th row vector of the sequence matrix (a single sequence)
                seq_vector = npmat_final_matrix[i, :]
                # Normalization.
                seq_vector_norm = min_max_norm(seq_vector, 0, self.global_dictionary.token_count)
                normalized_matrix.append(seq_vector_norm)
            npmat_final_matrix = numpy.array(normalized_matrix)
        return npmat_final_matrix