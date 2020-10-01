from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from interface.nn import NNModelInterface


class PyTorchNNMultiClassifier(NNModelInterface):
    """
    Implements a neural network in the context of multi-class classification.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: Callable) -> None:
        """

        Args:
            model: reference to a pytorch neural network
            optimizer: reference to a pytorch optimizer
            loss_function: reference to a callable loss function

        Returns: None
        """
        super().__init__(model)
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_function = loss_function

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the neural network.

        Args:
            model_input: 4 dimensional tensor: batch X RGB X H x W

        Returns: a tensor containing the model output
        """
        self.__model.eval()
        return self.__model(model_input)

    def interpret_output(self, output: torch.Tensor) -> tuple:
        """
        Interprets the model output.

        Args:
            output: the tensor to be interpreted

        Returns:

        """
        output = torch.nn.functional.softmax(output, dim=0)
        probabilities = np.max(output.detach().numpy(), axis=0)
        prob_argmax = np.argmax(output.detach().numpy(), axis=1)

        return prob_argmax, probabilities

    def backward(self, backward_step_input: tuple) -> None:
        """
        Only does a single backpropagation step and computes gradients.
        The gradients are not applied to the weights yet,
        which in pytorch is done by calling the step() function
        of the optimizer.

        Args:
            backward_step_input: a tuple in the form (model_output, labels)

        Returns: None

        """
        self.zero_grad()
        loss = self.__loss_function(backward_step_input[0], backward_step_input[1])
        loss.backward()

    def apply_gradients(self) -> None:
        """
        We don't really need this function to evaluate our model, it is
        only useful for training the model. For the sake of completeness
        though I include it.

        Returns: None

        """
        self.__optimizer.step()

    def gradient_for(self, model_input: tuple) -> torch.Tensor:
        """
        Computes the gradient for the model_input.
        Args:
            model_input:

        Returns: the gradients for the model_input

        """
        model_input[0].requires_grad = True
        labels = torch.from_numpy(model_input[1])

        output = self.forward(model_input[0])

        self.zero_grad()
        self.backward((output, labels))

        return model_input[0].grad.data

    def zero_grad(self) -> None:
        """
        Set the gradients to zero.

        Returns: None

        """
        self.__model.zero_grad()


def plot_img(img):
    np_img = img.detach().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
