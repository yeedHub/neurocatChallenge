from abc import ABCMeta, abstractmethod


class ModelInterface(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, model, params):
        """
        The constructor takes a reference to the model.
        The exact form of these references depend on the used framework.
        """
        pass
