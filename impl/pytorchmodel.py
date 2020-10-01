from collections import OrderedDict
from typing import Iterator

from torch.nn import Module, Parameter


class PytorchModel:
    """
    The PytorchModel class is responsible for initializing a pretrained pytorch model correctly.
    """

    def __init__(self, model: Module, params: dict) -> None:
        """

        Args:
            model: pytorch neural network module
            params: parameters for initializing the neural network correctly
        """
        self.__model = model

        # Load the pre-trained checkpoint
        checkpoint = params["checkpoint"]

        # The model was saved wrapped in torch.nn.DataParallel, therefore we have to strip the "module." prefix
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[len(params["remove_prefix"]):]
            new_state_dict[name] = v

        self.__model.load_state_dict(new_state_dict)

    def model(self) -> Module:
        """

        Returns: the initialized model

        """
        return self.__model

    def get_parameters(self) -> Iterator[Parameter]:
        """

        Returns: the model parameters that are trained (weights)

        """
        return self.__model.parameters()
