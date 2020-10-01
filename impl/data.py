import numpy as np
import torch

from interface.data import DataHandlerInterface


class PyTorchClassifierDataHandler(DataHandlerInterface):
    def __init__(self, dataset, params):
        self.__dataset = dataset[0]  # contains x and y_true
        self.__charlabels = dataset[1]  # human readable labels
        self.__loader = torch.utils.data.DataLoader(self.__dataset, **params)
        self.__dataiter = iter(self.__loader)

    # TODO: does not handle out of data request
    def get_next(self, count):
        samples = []
        labels = []

        for _ in range(count):
            x, y = self.__dataiter.next()

            if (isinstance(y, torch.Tensor)):
                y = y.detach().numpy()

            samples.append(x)
            labels.append(y)

        return (samples, np.array(labels))

    def size(self):
        return self.__dataset.__len__()

    def get_charlabels(self):
        return self.__charlabels
