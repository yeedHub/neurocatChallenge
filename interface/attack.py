from abc import ABCMeta, abstractmethod


class AttackInterface(metaclass=ABCMeta):
    """
    The AttackInterface implements an adversarial attack.
    """

    @abstractmethod
    def __init__(self, params):
        """

        Args:
            params: parameters of the attack

        Returns: None

        """
        self.__params = params

    @abstractmethod
    def __call__(self, attack_input):
        """
        Execute the attack.

        Args:
            attack_input: input needed to execute the attack

        Returns: None

        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        Set the parameters of the attack.

        Args:
            params: new parameters

        Returns: None

        """
        pass
