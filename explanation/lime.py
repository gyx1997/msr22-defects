import numpy
import scipy.sparse
import sklearn.metrics.pairwise
import math
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier
from explanation.base_explanation import BaseExplanator, BaseExplanation


def gaussian(d, w):
    return math.exp((-1 * d * d) / (w * w))


class LIME(BaseExplanator):
    """
    LIME Explanation for Bag of Atomic Features representation.
    For Details of LIME, see Ribeiro et al. in KDD'16.
    """

    def explanator_name(self):
        return "LIME"

    def __init__(self, **kwargs):
        """
        Constructor of LIME.

        :key num_samples: Number of perturbed samples to be generated. Default 1000.
        :key num_features: Number of features to be used in the local linear model. Default 5.
        :key tokens List of: tokens appeared globally.
        :key classifier: Black-box classifier, which must contains 'predict' function.
        """

        # Call parent's constructor.
        super(LIME, self).__init__(**kwargs)

        # Parse arguments.
        self.num_samples = kwargs.get("num_samples", 1000)
        self.num_features = kwargs.get("num_features", 5)

        # Define the field for explanation result.
        self.key_tokens = []

        # Define the private field for Bag Of Word (atomic feature) representation.
        self.__atomic_features = []

    def __build_atomic_features(self, x):
        """
        Build a list for atomic features by given instance x.

        :param x: The given instance which should be list of explanation.features.AtomicFeature.
        :rtype: None
        """
        for atomic_feature in x:
            if not self.__atomic_features.__contains__(atomic_feature):
                self.__atomic_features.append(atomic_feature)
        pass

    def __build_vector(self, x):
        """
        Generate the Bag-of-Token representation of the sample x to be explained.

        Args:
            x: The sample to be explained.
        Returns:
            A numpy array (vector) of sequence x with Bag-of-Token representation.
        """
        pass

    def __sample_around(self, x):
        """
        Method for sampling the perturbed instances.

        :param x: the instance to be explained.
        """

        num_samples = self.num_samples

        # Assume it always have the predict function.
        classify_func = self._predict

        sequence_length = len(x)

        # Generate the number of atomic features to be removed randomly.
        sample = numpy.random.randint(0, sequence_length, num_samples - 1)

        # Here matrix 'data' is formed as [[x_11, x_12, ..., x_1n],
        #                                  [x_21, x_22, ..., x_2n],
        #                                  ...
        #                                  [x_m1, x_m2, ..., x_mn]],
        # where x_ij is either 0 or 1, called feature selection sign,
        # which represents a corresponding atomic feature in position j is removed or
        # activated respectively. A row of matrix 'data' represents a perturbed sample.
        data = numpy.ones((num_samples, sequence_length))
        # The first row always means the original instance x to be explained.
        data[0] = numpy.ones(sequence_length)
        # 'Feature' here does not means the features used by black-box classifier.
        # It also does not mean an atomic feature in Bag-Of-Token representation.
        # It should be the existence of a given atomic feature in the token sequence.
        features_range = range(sequence_length)
        # Define the combined_data, which means Z[0] in the paper.
        perturbed_data = [x]
        for i, size in enumerate(sample, start=1):
            # Randomly inactive atomic features.
            inactive_features = numpy.random.choice(features_range, size, replace=False)
            # Mark the inactive features as zero.
            data[i, inactive_features] = 0
            # Try to make a perturbed sample by removing inactive feature(s).
            current_sequence = []
            for j in range(0, sequence_length):
                # If a single token is not marked as removed, it will be appended to the current
                # sample.
                if j not in inactive_features:
                    current_sequence.append(x[j])

            # Add the perturbed sample to set Z at line 4 of algorithm 1 in Ribeiro et al. in
            # KDD'16, i.e., Z[0] = Z[0] Union z'.
            perturbed_data.append(current_sequence)

        # Get the black-box classification result.
        labels = []
        for perturbed_sample in perturbed_data:
            # We do not feed the nested list directly since method BaseExplanation.__transform()
            # cannot process the nested class. Instead, we predict each perturbed sample and combined
            # them into the list labels. Note that for non-defect prediction, which prediction label
            # is 0, we modify to -1 to make regression model work. Otherwise, all of _coef will be 0
            # since the optimization object for regression model is MSE.
            prediction_label = classify_func(perturbed_sample)[0]
            labels.append([1 if prediction_label == 1 else -1])

        # Calculate the distance between all each generated sample and the original sample.
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(scipy.sparse.csr_matrix(data))[0] * 10
        return data, labels, distances

    def __feature_selection(self, data, labels, weights, num_features):
        ridge = Ridge(alpha=0, fit_intercept=True)
        used_features = []
        for _ in range(0, num_features):
            max_ = -10000000
            best_feature = 0
            for feature in range(0, data.shape[1]):
                if feature in used_features:
                    continue
                data_with_selected_features = data[:, used_features + [feature]]
                ridge.fit(data_with_selected_features, labels, sample_weight=weights)
                score = ridge.score(data_with_selected_features, labels, sample_weight=weights)
                if score > max_:
                    max_ = score
                    best_feature = feature
            used_features.append(best_feature)
        return numpy.array(used_features)

    def __build_bag_of_word_vector(self, original_sequence, flag_vector):
        """
        Reconstruct the perturbed sample(s) by original sample and the flag vector.

        :param original_sequence: The original sequence of the instance to be explained.
        :param flag_vector: Represents whether token(s) should appear in the reconstructed sequence.
        :returns: A python list which is reconstructed from original sequence and the flag vector.
        """

        # First, we build the sequence according to flag_vector.
        seq = []
        for i in range(0, len(original_sequence)):
            if flag_vector[i] != 0:
                seq.append(original_sequence[i])

        # Second, we build the dictionary to count the appearance of atomic features.
        feature_count = {}
        for atomic_feature in self.__atomic_features:
            feature_count[atomic_feature] = 0

        for atomic_feature in seq:
            feature_count[atomic_feature] += 1

        # Finally, we build the vector.
        result_vector = []
        for atomic_feature in self.__atomic_features:
            result_vector.append(feature_count[atomic_feature])

        return numpy.array(result_vector)

    def __calc_weights(self, distances):
        """
        Calculate the weights vector for different samples.

        :param distances: The distances for perturbed samples.
        :returns: A numpy vector which represents the weights after applying the kernel function.
        """
        weights = []
        for i in range(0, self.num_samples):
            weights.append(gaussian(distances[i], 2))
        return numpy.array(weights)

    def explain(self, X, **kwargs):
        """
        Explain a given prediction (also instance).

        :param X: A list of token sequence to be explained.
        :key linear_model: Specify the linear model for explanation.
             ridge_regressor, ridge_classifier, lasso accepted. Default ridge_regressor.
        :key verbose: Specify whether output the logging information.
        """

        num_features = self.num_features
        linear_model_type = kwargs.get("linear_model", "ridge_regressor")
        verbose = kwargs.get("verbose", True)

        # For different linear model type(s), build different regression models.
        linear = None
        if linear_model_type == "ridge_regressor":
            linear = Ridge(alpha=1, fit_intercept=True)
        elif linear_model_type == "ridge_classifier":
            # Note that attribute coef_ of Ridge Classifier is 2-dimension shape.
            linear = RidgeClassifier(alpha=1, fit_intercept=True)
        elif linear_model_type == "lasso":
            linear = Lasso(fit_intercept=True, alpha=1)
        if linear is None:
            raise ValueError("Parameter 'linear_model' must be one of ('ridge_regressor', 'ridge_classifier', 'lasso')")

        xid = 1
        result = []

        for x in X:

            if verbose:
                pass
                # print("Explaining %d/%d" % (xid, len(X)))

            # Before formally start the algorithm, we need to transform the instance object
            # into list of atomic features.
            x = x.features()

            # We build the list of atomic features first.
            self.__build_atomic_features(x)

            # Then Sample around the instance x.
            flag_matrix, labels, distances = self.__sample_around(x)

            # Then reconstruct the input for given instance with bag of word representation.
            matrix_perturbed_samples_bow = []

            for flag_vector in flag_matrix:
                # Build the bag-of-word vector of current perturbed sample.
                vec = self.__build_bag_of_word_vector(x, flag_vector)
                matrix_perturbed_samples_bow.append(vec)

            # Convert the python 2-dimension array into numpy 2-dimension matrix.
            npmat_perturbed_samples_bow = numpy.array(matrix_perturbed_samples_bow)
            npvec_weights = self.__calc_weights(distances)
            used_features = self.__feature_selection(npmat_perturbed_samples_bow, labels, npvec_weights, num_features)
            # Build simple model.
            linear.fit(npmat_perturbed_samples_bow[:, used_features], labels, sample_weight=npvec_weights)
            # Get the coef vector. Note that attribute _coef of RidgeClassifier is 2-dimension.
            npvec_coef = linear.coef_[0]  # if linear_model_type == "ridge_classifier" else linear.coef_
            # Build the result.
            key_features = []
            for i in range(0, num_features):
                # Note that even LIME can only return the atomic features, we still make it as list here, to keep the
                # consistency with other explanation methods, since they may return features with continuous atomic
                # features.
                key_features.append(([self.__atomic_features[used_features[i]]], npvec_coef[i]))

            # Sort the key features and add them into the result. Here we use the probability itself
            # instead of using its absolute value, since in defect prediction we usually focus more
            # on the defective instances, i.e., we are not interested in the atomic features (tokens)
            # which lead to the clean prediction.
            result.append(LIME.Explanation(sorted(key_features, key=lambda _: abs(_[1]), reverse=True)))

            # Increase the index for verbose logging info.
            xid += 1

        return result

    class Explanation(BaseExplanation):
        """
        Class of LIME explanation.
        """

        def __init__(self, key_features):
            """
            Constructor of LIME explanation.
            :param key_features: The key features returned by LIME.
            """
            super().__init__(key_features=key_features)
