from abc import ABCMeta, abstractmethod


class AttackInterface(metaclass=ABCMeta):
    """
    The AttachInterface implements an adversarial attack.
    """

    @abstractmethod
    def __call__(self):
        """
        Execute the attack.

        Returns: optionally returns the result of the attack.

        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        Set the parameters of the attack.
        Args:
            params: New parameters.

        """
        pass
