import copy
import numpy
from models.resampling import BaseResampler


class OversamplingResampler(BaseResampler):
    """
    An simple implementation of oversampling for code token sequences.
    """
    def fit_resample(self, X, y):
        """
        Resampling the training data by oversampling.

        :param X: The code token sequences.
        :param y: The labels.
        :return: A tuple of X, y which are resampled data and labels.
        """
        X_train = copy.deepcopy(X)
        y_train = copy.deepcopy(y)
        total_len = len(X_train)
        count_array = [0, 0]
        instance_array = [[], []]
        extend_instances = []

        for i in range(0, total_len):
            count_array[y[i]] += 1
            instance_array[y[i]].append(X_train[i])

        if count_array[0] == count_array[1]:
            pass
        else:
            # Detect the major and minor class.
            minor_class = 0 if count_array[1] > count_array[0] else 1
            major_class = 1 - minor_class
            # Do the oversampling process.
            while count_array[major_class] - count_array[minor_class] >= count_array[minor_class]:
                count_array[minor_class] += count_array[minor_class]
                extend_instances.extend(instance_array[minor_class])
            if count_array[minor_class] < count_array[major_class]:
                extend_instances.extend(
                    numpy.random.choice(instance_array[minor_class],
                                        count_array[major_class] - count_array[minor_class])
                )
            # Building resampled X and y (labels).
            X_train.extend(extend_instances)
            y_train = numpy.concatenate(
                (y_train, numpy.array([minor_class for _ in range(0, len(X_train) - len(y_train))]))
            )

        return X_train, y_train
