import numpy as np
from torch import Tensor
from torch import clamp

from interface.attack import AttackInterface


class FGSMAttack(AttackInterface):
    """
    Implementation of the fast gradient sign method by Goodfellow et al.
    """

    def __init__(self, params: dict = None) -> None:
        """

        Args:
            params: parameters for the fast gradient sign method attack

        Returns: None

        """
        super().__init__(params)
        if params is None:
            params = {"eps": 0.07}
        self.__params = params

    def __call__(self, attack_input: tuple) -> Tensor:
        """
        Execute the attack.

        Args:
            attack_input: input to the attack (batch, gradient)

        Returns: the attacked samples

        """
        perturbed = attack_input[0] + self.__params["eps"] * attack_input[1].sign()
        perturbed = clamp(perturbed, np.min(attack_input[0].detach().numpy()), np.max(attack_input[0].detach().numpy()))

        return perturbed

    def set_params(self, params: dict) -> None:
        """
        Set the attack parameters.

        Args:
            params: new parameters

        Returns: None

        """
        self.__params = params
