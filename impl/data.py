import numpy as np
import torch
from torch.utils.data import Dataset

from interface.data import DataHandlerInterface


class PyTorchClassifierDataHandler(DataHandlerInterface):
    """
    Implementation of a DataHandler for classification with pytorch.
    """

    def __init__(self, dataset: tuple, params: dict) -> None:
        """

        Args:
            dataset: the dataset in pytorch format
            params: dictionary containing parameters for the data loader

        Returns: None

        """
        super().__init__(dataset, params)
        self.__data = dataset[0]  # contains x and y_true
        self.__char_labels = dataset[1]  # human readable labels
        self.__loader = torch.utils.data.DataLoader(self.__data, **params)
        self.__data_iterator = iter(self.__loader)

    def get_next(self, count: int) -> tuple:
        """
        Get the next count batches.

        Args:
            count: number of batches

        Returns: tuple in the form (batches, labels)

        """
        samples = []
        labels = []

        for _ in range(count):
            x, y = self.__data_iterator.next()

            if isinstance(y, torch.Tensor):
                y = y.detach().numpy()

            samples.append(x)
            labels.append(y)

        return samples, np.array(labels)

    def size(self) -> int:
        """

        Returns: size of the dataset

        """
        return self.__data.__len__()

    def get_char_labels(self) -> list:
        """

        Returns: list of human readable character labels

        """
        return self.__char_labels
