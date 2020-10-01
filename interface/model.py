from abc import ABCMeta, abstractmethod


class ModelInterface(metaclass=ABCMeta):
    """
    The ModelInterface can be used to generalize different model types.
    """

    @abstractmethod
    def __init__(self, model):
        """

        Args:
            model: reference to the model (as represented by a framework for example)

        Returns: None

        """
        self.__model = model
        pass
