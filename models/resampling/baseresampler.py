from abc import ABCMeta, abstractmethod


class BaseResampler(metaclass=ABCMeta):
    @abstractmethod
    def fit_resample(self, X, y):
        pass
