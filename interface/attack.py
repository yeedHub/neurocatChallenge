from abc import ABCMeta, abstractmethod


class AttackInterface(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        """
        Execute the attack.

        Returns: optionally returns the result of the attack.

        """
        pass
