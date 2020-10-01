from abc import ABCMeta, abstractmethod


class ModelAnalysisInterface(metaclass=ABCMeta):
    """
    The ModelAnalysisInterface calculates different metrics for a given model.
    """

    @abstractmethod
    def __init__(self):
        """

        Returns: None

        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the calculated/collected metrics.

        Returns: None

        """
        pass

    @abstractmethod
    def __call__(self, model, model_input):
        """
        Do the analysis.

        Args:
            model: reference to the model
            model_input: input to the model

        Returns: None

        """
        pass
