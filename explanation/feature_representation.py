"""
Class definitions for feature representation.
"""
import copy


class AtomicFeature:
    """
    Atomic Feature(s). It may be a single token (string), or a list of adjacent tokens.
        However, the internal representation of atomic feature is a list of token(s).
        If token level is applied, the list contains only 1 token (string).
    """

    def __init__(self, data, token_lines=0):
        """
        Constructor of AtomicFeature.

        :param data: A list of tokens (string).
        """
        if (not isinstance(data, list)) \
                or (len(data) == 0) \
                or (not isinstance(data[0], str)):
            raise TypeError("AtomicFeature must be initialized with non-empty list of string (tokens).")

        self.tokens = data
        self.token_lines = token_lines

    def __hash__(self):
        """
        Define the __hash__() operation to make it possible to be used in hash table-based
            data structure such as dictionary.
        """

        # For list of tokens, we concatenate all tokens, and return the hash of combined string.
        return hash("".join(self.tokens))

    def __eq__(self, other):
        """
        Define the __eq__() operator to make it possible to be used in hash table-based
            data structure such as dictionary.
        """

        # Note that if the length of compared object is not equals to the length of this object,
        # False is automatically returned.
        if len(self.tokens) != len(other.tokens):
            return False

        # For list of tokens, we compare each token, and return True if all tokens are the same.
        # Otherwise, False returned.
        for i in range(0, len(self.tokens)):
            if self.tokens[i] != other.tokens[i]:
                return False

        return True

    def __radd__(self, other):
        return other + self

    def __add__(self, other):
        """
        Define the __add__() operator to make it easier for combination.
        """
        if not isinstance(other, AtomicFeature):
            raise TypeError("Only AtomicFeatures are able to be added.")

        new_data = copy.deepcopy(self.tokens)
        new_data.extend(other.tokens)
        return AtomicFeature(new_data)

    def __str__(self):
        """
        Define __str__() operation.
        """
        str_data = ", ".join(self.tokens)
        return "[%s]" % str_data


class Instance:
    """
    Instance with representation using atomic features.
    """

    class FeatureIterator:
        """
        Iterator for atomic features.
        """

        def __init__(self, instance_object):
            self._instance_object = instance_object
            self._pointer = 0

        def __next__(self):
            """
            Implementation of __next__ for making it iterable.
            """
            if self._pointer > len(self._instance_object):
                raise StopIteration()
            feature = self._instance_object[self._pointer]
            self._pointer += 1
            return feature

    def __init__(self, X, lines):
        self._atomic_features = []
        self._atomic_features_count = {}
        self._atomic_features_mapper = {}
        self._atomic_features_sequence = []
        self._length = 0
        self._lines = lines

        for i in range(0, len(X)):
            raw_atomic_feature = X[i]
            # Iterate all raw atomic features (represented by list of tokens) to build
            # atomic_feature.
            atomic_feature = AtomicFeature(raw_atomic_feature, self._lines[i])
            if self._atomic_features_mapper.__contains__(atomic_feature):
                # In this situation, the current atomic feature has been added into
                # self._atomic_features. Hence, we simply increase its counter by 1.
                self._atomic_features_count[atomic_feature] += 1
            else:
                # Otherwise, we added it into self._atomic_features.
                # Save the feature id.
                feature_id = len(self._atomic_features)
                # Update the data structure.
                self._atomic_features.append(atomic_feature)
                self._atomic_features_count[atomic_feature] = 1
                self._atomic_features_mapper[atomic_feature] = feature_id

            # Finally, we build the sequence by using the reference.
            self._atomic_features_sequence.append(self._atomic_features_mapper[atomic_feature])

        # Check the consistency on lengths.
        assert (len(self._atomic_features) == len(self._atomic_features_mapper))
        assert (len(self._atomic_features) == len(self._atomic_features_count))

        # Cache the lengths.
        self._length = len(self._atomic_features)

    def __getitem__(self, item):
        """
        Returns an atomic feature by given subscript or slice. To make it consistency,
            only list will be returned.
        """
        
        if len(self._atomic_features_sequence) == 0:
            return []
        
        if isinstance(item, int):
            # Got int as subscript.
            return [self._atomic_features[self._atomic_features_sequence[item]]]
        else:
            # Got Slice or list.
            result = []
            selected_indices = self._atomic_features_sequence[item]
            for item_id in selected_indices:
                result.append(self._atomic_features[item_id])
            return result

    def __iter__(self):
        """
        Returns the iterator of atomic features (ordered as the raw sequence).
        """
        return Instance.FeatureIterator(self)

    def __len__(self):
        """
        Returns the length of current instance.
        """
        return self._length

    def features(self, **kwargs):
        """
        Returns features by given interval.

        :key start: Optional. The index of start feature. Default 0.
        :key end: Optional. The index of end feature. Default the index of last feature.
        :return: Returns features by given interval.
        """
        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(self._atomic_features_sequence))
        if start == end:
            return self[start]
        return self[start:end]


def features_to_tokens(features):
    """
    Method to convert list of atomic features into a list of tokens.

    :param features: List of atomic features.
    :return: A list of tokens.
    """
    result = []
    for feature in features:
        result.extend(feature.tokens)
    return result
