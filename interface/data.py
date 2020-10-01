from abc import ABCMeta, abstractmethod


class DataHandlerInterface(metaclass=ABCMeta):
    """
    The DataHandlerInterface generalizes the data handling of different frameworks
    or other means of loading data.
    """

    @abstractmethod
    def __init__(self, dataset, params):
        """

        Args:
            dataset: the dataset
            params: parameters for loading data

        Returns: None

        """
        self.__dataset = dataset
        self.__params = params
        pass

    @abstractmethod
    def get_next(self, count):
        """
        Depending on how the (frameworks) data pipeline is configured this
        function returns count number of samples/batches.

        Args:
            count: number of requested samples/batches

        Returns: requested samples/batches

        """
        pass

    @abstractmethod
    def size(self):
        """

        Returns: the number of data points in the dataset

        """
        pass
