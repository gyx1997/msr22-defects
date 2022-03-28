import explanation.feature_representation
from explanation.base_explanation import BaseExplanator, BaseExplanation
from explanation.feature_representation import features_to_tokens
from utils.out import Out
from explanation.utils import find
import numpy as np


class RandomSelection(BaseExplanator):

    def explanator_name(self):
        return "Random"

    def __init__(self, **kwargs):
        # Call parent's constructor.
        super(RandomSelection, self).__init__(**kwargs)
        self.num_features = kwargs.get("num_features", 5)
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
            x_unique_features = list(get_unique_features(x))
            feature_scores = []
            selected_features = np.random.choice(x_unique_features,
                                                 min(len(x_unique_features), self.num_features),
                                                 replace=False)
            for selected_feature in selected_features:
                proba = self.classifier.predict_proba([features_to_tokens([selected_feature])])[0][self._golden_label]
                feature_scores.append(([selected_feature], proba))

            feature_scores = sorted(feature_scores, key=lambda _: _[1], reverse=True)
            result.append(RandomSelection.Explanation(feature_scores[:min(len(x_unique_features), self.num_features)]))

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
