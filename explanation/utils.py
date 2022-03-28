import explanation.feature_representation


def find(instance, feature):
    """
    Find the appearances of a feature in a given instance.

    :param instance:
    :param feature:
    :return:
    """

    if not isinstance(instance, explanation.feature_representation.Instance):
        raise TypeError("Argument instance must be type of "
                        "explanation.feature_representation.Instance.")

    raw_features = instance.features()
    result = []
    for i in range(0, len(raw_features)):
        offset = 0
        while i + offset < len(raw_features) and offset < len(feature):
            if raw_features[i + offset] != feature[offset]:
                break
            offset += 1
        if offset == len(feature):
            result.append(i)
    return result
