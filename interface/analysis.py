from abc import ABCMeta, abstractmethod


class ModelAnalysisInterface(metaclass=ABCMeta):
    """
    The ModelAnalysisInterface calculates different metrics for a given model.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the calculated/collected metrics.

        Returns: None

        """
        pass

    @abstractmethod
    def __call__(self, model, input, params=None):
        """

        Args:
            model: Instance of interface.ModelInterface
            X: List of inputs (samples or batches) to model.

        Returns: Dictionary with the calculated metrics.

        """
        pass
