from abc import ABCMeta, abstractmethod


class DataHandlerInterface(metaclass=ABCMeta):
    """
    The DataHandlerInterface generalizes the data handling of different
    frameworks.
    """

    @abstractmethod
    def __init__(self, dataset, params):
        pass

    @abstractmethod
    def get_next(self, count):
        """
        Depending on how the frameworks data pipeline is configured this
        function returns count number of samples/batches.

        Args:
            count: How many samples/batches are requested.

        Returns: count number of samples/batches.

        """
        pass

    @abstractmethod
    def size(self):
        """

        Returns: The number of samples in the dataset.

        """
        pass
