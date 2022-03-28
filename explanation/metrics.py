from explanation.feature_representation import features_to_tokens, AtomicFeature
from explanation.utils import find
from utils.out import Out


def aopc_score(samples, explanations, classifier, **kwargs):
    """
    Calculate the AOPC (Area Over the Perturbation Curve) score for the explanation.

    :param explanations: Required. The explanations of given samples. Should be list of
        BaseExplanation, and attribute key_features should be available.
    :param samples: Required. The predicted samples to be explained. Should be list of
        feature_representation.Instance.
    :param classifier: Required. The black-box classifier. Method 'predict_proba' required.

    :key golden_label: Optional. The index of golden label. Default is 1 (i.e., defective
        in defect prediction tasks).
    :key score_threshold: Optional. The threshold to determine whether the feature contributes
        to the prediction. It should be determined by the explanator. Default is 0.
    :return: The AOPC score.
    """

    # We should check the consistency of dimensions of samples and explanations.
    if len(samples) != len(explanations):
        raise ValueError("Argument 'samples' and 'explanations' must have the same length.")

    golden_label = kwargs.get("golden_label", 1)
    score_threshold = kwargs.get("score_threshold", 0)

    # Define sum of AOPC for all instances (samples).
    sum = 0

    # Define the counter for shifting.
    shifting_num = 0

    # Iterate all samples and calculate the difference of probabilities.
    for i in range(0, len(samples)):
        # Pick up the current sample and convert the Instance into list of atomic features.
        x = samples[i].features()

        # Pick up its key features in explanation. Note that a key feature is represented as a
        # tuple (AtomicFeature, float). The first element is the atomic feature itself, while
        # the second one is the weight.
        key_features = []
        for feature, weight in explanations[i].key_features:
            # Only pick the features which contribute to the prediction.
            if golden_label == 1:
                # Golden label is 1 (also means defective).
                if weight > score_threshold:
                    key_features.append(feature)
            else:
                # Golden label is 0 (also means non-defective).
                if weight < score_threshold:
                    key_features.append(feature)

        # For picked sample, we calculate its AOPC score.
        # First, we build the perturbed input.
        x_raw = x
        x_perturbed = []

        binary_flag = [1 for _ in range(0, len(x))]

        for key_feature in key_features:
            appearances = find(samples[i], key_feature)
            if len(appearances) > 0:
                for appearance in appearances:
                    for j in range(appearance, appearance + len(key_feature)):
                        binary_flag[j] = 0

        for j in range(0, len(binary_flag)):
            if binary_flag[j] > 0:
                x_perturbed.append(x_raw[j])

        Out.write("    Num of atomic features (raw, perturbed) "
                  + str(len(x_raw)) + ", " + str(len(x_perturbed)))
        Out.write("    Num of code entities (raw, perturbed) "
                  + str(len(features_to_tokens(x_raw))) + ", " + str(len(features_to_tokens(x_perturbed))))

        # Then, we feed the perturbed input sample into the classifier.
        raw_proba = classifier.predict_proba([features_to_tokens(x_raw)])[0][golden_label]
        perturbed_proba = classifier.predict_proba([features_to_tokens(x_perturbed)])[0][golden_label]
        if (raw_proba >= 0.5 and perturbed_proba < 0.5) or (raw_proba <= 0.5 and perturbed_proba > 0.5):
            shifting_num += 1
        Out.write("   " + str(raw_proba) + ", " + str(perturbed_proba))
        sum += raw_proba - perturbed_proba

    return sum / len(samples), shifting_num / len(samples)


def deletion_metrics(samples, explanations, classifier, **kwargs):
    """
    Calculate the deletion-based score for the explanation.

    :param explanations: Required. The explanations of given samples. Should be list of
        BaseExplanation, and attribute key_features should be available.
    :param samples: Required. The predicted samples to be explained. Should be list of
        feature_representation.Instance.
    :param classifier: Required. The black-box classifier. Method 'predict_proba' required.

    :key metrics: Optional. The metrics to be calculated. Supports 'aopc' and 'fdf'. Default
        is ['aopc']
    :key golden_label: Optional. The index of golden label. Default is 1 (i.e., defective
        in defect prediction tasks).
    :key score_threshold: Optional. The threshold to determine whether the feature contributes
        to the prediction. It should be determined by the explanator. Default is 0.
    :key verbose: Optional. The flag for outputing the debug information. Default is True.
    :return: The AOPC score.
    """

    # Check the consistency of dimensions of samples and explanations.
    if len(samples) != len(explanations):
        raise ValueError("Argument 'samples' and 'explanations' must have the same length.")

    metrics = kwargs.get("metrics", ['aopc'])
    golden_label = kwargs.get("golden_label", 1)
    score_threshold = kwargs.get("score_threshold", 0)
    verbose = kwargs.get("verbose", True)

    proba_sum = 0  # Define sum of AOPC for all instances (samples).
    shifting_num = 0  # Define the counter for shifting.

    # Iterate all samples and calculate the difference of probabilities.
    for i in range(0, len(samples)):
        # Pick up the current sample and convert the Instance into list of atomic features.
        x = samples[i].features()

        # Pick up its key features in explanation. Note that a key feature is represented as a
        # tuple (AtomicFeature, float). The first element is the atomic feature itself, while
        # the second one is the weight.
        key_features = []
        for feature, weight in explanations[i].key_features:
            # Only pick the features which contribute to the prediction.
            if golden_label == 1:
                # Golden label is 1 (also means defective).
                if weight > score_threshold:
                    key_features.append(feature)
            else:
                # Golden label is 0 (also means non-defective).
                if weight < score_threshold:
                    key_features.append(feature)

        # For picked sample, we calculate its AOPC score.
        # First, we build the perturbed input.
        x_raw = x
        x_perturbed = []

        binary_flag = [1 for _ in range(0, len(x))]

        for key_feature in key_features:
            appearances = find(samples[i], key_feature)
            if len(appearances) > 0:
                for appearance in appearances:
                    for j in range(appearance, appearance + len(key_feature)):
                        binary_flag[j] = 0

        for j in range(0, len(binary_flag)):
            if binary_flag[j] > 0:
                x_perturbed.append(x_raw[j])

        # Then, we feed the perturbed input sample into the classifier.
        raw_proba = classifier.predict_proba([features_to_tokens(x_raw)])[0][golden_label]
        perturbed_proba = classifier.predict_proba([features_to_tokens(x_perturbed)])[0][golden_label]
        if (raw_proba >= 0.5 and perturbed_proba < 0.5) or (raw_proba <= 0.5 and perturbed_proba > 0.5):
            shifting_num += 1

        if verbose:
            Out.write("For Instance %d," % i)
            Out.write("The numbers of atomic features (raw, perturbed) are "
                      + str(len(x_raw)) + ", " + str(len(x_perturbed)) + ";")
            Out.write("The numbers of code entities (raw, perturbed) are "
                      + str(len(features_to_tokens(x_raw))) + ", " + str(len(features_to_tokens(x_perturbed))) + ";")
            Out.write("Probability before and after perturbation are " + str(raw_proba) + ", " + str(perturbed_proba) + ".")
        proba_sum += raw_proba - perturbed_proba

    # Get results.
    result = {}
    if 'aopc' in metrics:
        result['aopc'] = proba_sum / len(samples)
    if 'fdf' in metrics:
        result['fdf'] = shifting_num / len(samples)
    return result


def explanation_precision(samples, explanations, classifier, **kwargs):
    """

    :param samples:
    :param explanations:
    :param classifier:
    :param kwargs:
    :return:
    """
    # We should check the consistency of dimensions of samples and explanations.
    if len(samples) != len(explanations):
        raise ValueError("Argument 'samples' and 'explanations' must have the same length.")

    golden_label = kwargs.get("golden_label", 1)
    score_threshold = kwargs.get("score_threshold", 0)

    sum = 0  # Sum of explanation precision for all predictions.

    # Iterate all samples and make predictions.
    for i in range(0, len(samples)):
        sample_instance = samples[i]
        sample_explanation = explanations[i]

        # Pick up the current sample and convert the Instance into list of atomic features.
        x = samples[i].features()

        # Pick up its key features in explanation. Note that a key feature is represented as a
        # tuple (AtomicFeature, float). The first element is the atomic feature itself, while
        # the second one is the weight.
        key_features = []

        for feature, weight in sample_explanation.key_features:
            # Only pick the features which contribute to the prediction.
            if golden_label == 1:
                # Golden label is 1 (also means defective).
                if weight > score_threshold:
                    key_features.append(feature)
            else:
                # Golden label is 0 (also means non-defective).
                if weight < score_threshold:
                    key_features.append(feature)

        def atomic2str(atomic_features):
            """
            Helper function to convert atomic feature to string tokens.
            """
            str_list = []
            for af in atomic_features:
                str_list.append(str(af))
            feat_str = ", ".join(str_list)
            return feat_str

        # Construct new sequence with explained features padded.
        appearances = []
        original_sequence = features_to_tokens(sample_instance.features())

        # First we need to find the appearances of features in explanation.
        for feature in key_features:
            find_result = find(sample_instance, feature)
            if len(find_result) > 0:
                # Find the feature, added it into appearence list.
                appearances.append((atomic2str(feature), find_result))

        # Then, we build the new sequence.
        padded_sequence = []
        for j in range(0, len(original_sequence)):
            flag = False
            for tok, appear_pos in appearances:
                if j in appear_pos:
                    # The feature appears in explanation. Thus, just copy it to the new sequence.
                    flag = True
                    padded_sequence.append(tok[1:-1])
                    break
            if not flag:
                # Otherwise, the feature is less important according to the explanation.
                # Thus, we replace it with special token "<PADDING>".
                padded_sequence.append("<PADDING>")

        # Finally, we calculate I(s[i]) - I(s_Expl[i]).
        sum += classifier.predict([padded_sequence])[0]

    return sum / len(samples)


def aopc_padding_score(samples, explanations, classifier, **kwargs):
    """
    Calculate the AOPC (Area Over the Perturbation Curve) score for the explanation.

    :param explanations: Required. The explanations of given samples. Should be list of
        BaseExplanation, and attribute key_features should be available.
    :param samples: Required. The predicted samples to be explained. Should be list of
        feature_representation.Instance.
    :param classifier: Required. The black-box classifier. Method 'predict_proba' required.

    :key golden_label: Optional. The index of golden label. Default is 1 (i.e., defective
        in defect prediction tasks).
    :key score_threshold: Optional. The threshold to determine whether the feature contributes
        to the prediction. It should be determined by the explanator. Default is 0.
    :return: The AOPC score.
    """

    # We should check the consistency of dimensions of samples and explanations.
    if len(samples) != len(explanations):
        raise ValueError("Argument 'samples' and 'explanations' must have the same length.")

    golden_label = kwargs.get("golden_label", 1)
    score_threshold = kwargs.get("score_threshold", 0)

    # Define sum of AOPC for all instances (samples).
    sum = 0

    # Define the counter for shifting.
    shifting_num = 0

    # Iterate all samples and calculate the difference of probabilities.
    for i in range(0, len(samples)):
        # Pick up the current sample and convert the Instance into list of atomic features.
        x = samples[i].features()

        # Pick up its key features in explanation. Note that a key feature is represented as a
        # tuple (AtomicFeature, float). The first element is the atomic feature itself, while
        # the second one is the weight.
        key_features = []
        for feature, weight in explanations[i].key_features:
            # Only pick the features which contribute to the prediction.
            if golden_label == 1:
                # Golden label is 1 (also means defective).
                if weight > score_threshold:
                    key_features.append(feature)
            else:
                # Golden label is 0 (also means non-defective).
                if weight < score_threshold:
                    key_features.append(feature)

        # For picked sample, we calculate its AOPC score.
        # First, we build the perturbed input.
        x_raw = x
        x_perturbed = []

        binary_flag = [1 for _ in range(0, len(x))]

        for key_feature in key_features:
            appearances = find(samples[i], key_feature)
            if len(appearances) > 0:
                for appearance in appearances:
                    for j in range(appearance, appearance + len(key_feature)):
                        binary_flag[j] = 0

        for j in range(0, len(binary_flag)):
            if binary_flag[j] > 0:
                x_perturbed.append(x_raw[j])
            else:
                padding_feature = AtomicFeature(["<PADDING>"], 0)
                x_perturbed.append(padding_feature)

        Out.write("    Num of atomic features (raw, perturbed) "
                  + str(len(x_raw)) + ", " + str(len(x_perturbed)))
        Out.write("    Num of code entities (raw, perturbed) "
                  + str(len(features_to_tokens(x_raw))) + ", " + str(len(features_to_tokens(x_perturbed))))

        # Then, we feed the perturbed input sample into the classifier.
        raw_proba = classifier.predict_proba([features_to_tokens(x_raw)])[0][golden_label]
        perturbed_proba = classifier.predict_proba([features_to_tokens(x_perturbed)])[0][golden_label]
        if (raw_proba >= 0.5 and perturbed_proba < 0.5) or (raw_proba <= 0.5 and perturbed_proba > 0.5):
            shifting_num += 1
        Out.write("   " + str(raw_proba) + ", " + str(perturbed_proba))
        sum += raw_proba - perturbed_proba

    return sum / len(samples), shifting_num / len(samples)
