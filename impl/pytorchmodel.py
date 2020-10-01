from interface.model import ModelInterface

from collections import OrderedDict

class PytorchModel(ModelInterface):
    def __init__(self, model, params):
        self.__model = model

        # Load the pre-trained checkpoint
        checkpoint = params["checkpoint"]

        # The model was saved wrapped in torch.nn.DataParallel, therefore we have to strip the "module." prefix
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[len(params["remove_prefix"]):]
            new_state_dict[name] = v

        self.__model.load_state_dict(new_state_dict)

    def model(self):
        return self.__model

    def get_parameters(self):
        """

        Returns: The model parameters, that are trained.

        """
        return self.__model.parameters()