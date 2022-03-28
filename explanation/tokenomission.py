import math

import explanation.feature_representation
from explanation.base_explanation import BaseExplanator, BaseExplanation
from explanation.feature_representation import features_to_tokens
from utils.out import Out
from explanation.utils import find


class TokenOmission(BaseExplanator):

    def explanator_name(self):
        return "Word Omission"

    def __init__(self, **kwargs):
        # Call parent's constructor.
        super(TokenOmission, self).__init__(**kwargs)

        self.num_features = kwargs.get("num_features", "auto")
        self.max_features_percent = kwargs.get("max_features_percent", 0.5)
        self._golden_label = kwargs.get("golden_label", 1)

        pass

    def explain(self, X, **kwargs):
        """
        Explain a given prediction (also instance).

        :param X: A list of token sequence to be explained.
        :key verbose: Specify whether output the logging information.
        """

        verbose = kwargs.get("verbose", False)

        def get_unique_features(instance):
            unique_features = set()
            for feature in instance.features():
                unique_features.add(feature)
            return unique_features

        result = []

        for x in X:
            x_unique_features = get_unique_features(x)
            num_unique_features = 0
            max_unique_features = math.ceil(self.max_features_percent * len(x_unique_features)) if self.num_features == "auto" \
                else self.num_features
            x_instance = x
            x = x.features()
            feature_scores = []

            for unique_feature in x_unique_features:
                x_perturbed = []
                for feature in x:
                    if feature == unique_feature:
                        continue
                    x_perturbed.append(feature)

                proba1 = self.classifier.predict_proba([features_to_tokens(x)])[0][self._golden_label]
                proba2 = self.classifier.predict_proba([features_to_tokens(x_perturbed)])[0][self._golden_label]

                feature_scores.append(([unique_feature], proba1 - proba2))

            feature_scores = sorted(feature_scores, key=lambda _: _[1], reverse=True)

            key_features = []
            if self.num_features == "auto":
                for feat in feature_scores:
                    if feat[1] <= 0.001:
                        break

                    x_raw = x
                    x_perturbed = []

                    binary_flag = [1 for _ in range(0, len(x))]

                    for key_feature in key_features:
                        appearances = find(x_instance, key_feature)
                        if len(appearances) > 0:
                            for appearance in appearances:
                                for j in range(appearance, appearance + len(key_feature)):
                                    binary_flag[j] = 0

                    for j in range(0, len(binary_flag)):
                        if binary_flag[j] > 0:
                            x_perturbed.append(x_raw[j])

                    # Then, we feed the perturbed input sample into the classifier.
                    raw_proba = self.classifier.predict_proba([features_to_tokens(x_raw)])[0][self._golden_label]
                    perturbed_proba = self.classifier.predict_proba([features_to_tokens(x_perturbed)])[0][self._golden_label]

                    if (raw_proba >= 0.5 and perturbed_proba < 0.5) or \
                            (raw_proba <= 0.5 and perturbed_proba > 0.5) or \
                            len(key_features) > max_unique_features:
                        break

                    key_features.append(feat[0])

                result.append(TokenOmission.Explanation(feature_scores[:len(key_features)]))
            else:
                result.append(TokenOmission.Explanation(feature_scores[:min(len(x_unique_features), self.num_features)]))

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
