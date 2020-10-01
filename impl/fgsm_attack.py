import numpy as np
import torch

from interface.attack import AttackInterface


class FGSMAttack(AttackInterface):
    def __init__(self, params=None):
        if params:
            self.__params = params
        else:
            self.__params = {"eps": 0.07}

    def __call__(self, x, grad):
        perturbed = x + self.__params["eps"] * grad.sign()
        perturbed = torch.clamp(perturbed, np.min(x.detach().numpy()), np.max(x.detach().numpy()))

        return perturbed

    def set_params(self, params):
        self.__params = params
