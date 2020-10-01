from abc import ABCMeta, abstractmethod


class ModelInterface(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, model, params):
        """
        The constructor takes a reference to the model, and params
        to initialize the model correctly.
        """
        pass

    @abstractmethod
    def model(self):
        """

        Returns: the correctly initialized model.

        """
        pass