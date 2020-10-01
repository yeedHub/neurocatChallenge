from abc import abstractmethod

from interface.model import ModelInterface


class NNModelInterface(ModelInterface):
    """
    The NNModelInterface generalizes neural network models.
    """

    @abstractmethod
    def forward(self, model_input):
        """
        Forward step of the neural network model.

        Args:
            model_input: input to the model

        Returns: output of the neural network

        """
        pass

    @abstractmethod
    def interpret_output(self, output):
        """
        Interpret the output of self.forward(model_input). This can be useful e.g. when we build a
        multi-class classifier and our models output needs to be normalized to probabilities.

        Args:
            output: output of the call to forward(model_input)

        Returns: interpreted output

        """
        pass

    @abstractmethod
    def backward(self, backward_step_input):
        """
        Backward step of the neural network model. backward_step_input has to be the direct model output,
        before calling self.interpret_output(output).

        Args:
            backward_step_input: direct model output

        Returns: None

        """
        pass

    @abstractmethod
    def apply_gradients(self):
        """
        Applies the gradients to the weights of the model. This function should be called
        after self.backward(backward_step_input).

        Returns: None

        """
        pass

    @abstractmethod
    def zero_grad(self):
        """
        Set all gradients to zero.

        Returns: None

        """
        pass

    @abstractmethod
    def gradient_for(self, model_input):
        """
        Calculate the gradient for model_input.

        Returns: the calculated gradient

        """
        pass
