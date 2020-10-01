import matplotlib.pyplot as plt
import numpy as np
import torch

from interface.nn import NNModelInterface


class PyTorchNNMulticlassifier(NNModelInterface):
    def __init__(self, model, optimizer, loss_function):
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_function = loss_function

    def forward(self, x):
        """
        x has to be 4 dimensional: batch X RGB X H x W
        """
        self.__model.eval()
        return self.__model(x)

    def interpret_output(self, output):
        output = torch.nn.functional.softmax(output, dim=0)
        probabilities = np.max(output.detach().numpy(), axis=0)
        probargmax = np.argmax(output.detach().numpy(), axis=1)

        return (probargmax, probabilities)

    def backward(self, output, ytrue):
        """
        Only does a single backpropagation step and computes gradients.
        The gradients are not applied to the weights yet,
        which in pytorch is done by calling the step() function
        of the optimizer.
        """
        self.zero_grad()
        loss = self.__loss_function(output, ytrue)
        loss.backward()

    def apply_gradients(self):
        """
        We don't really need this function to evaluate our model, it is
        only useful for training the model. For the sake of completeness
        though I include it.
        """
        self.__optimizer.step()

    def gradient_for(self, x, ytrue):
        x.requires_grad = True
        ytrue = torch.from_numpy(ytrue)

        output = self.forward(x)

        self.zero_grad()
        self.backward(output, ytrue)

        return x.grad.data

    def zero_grad(self):
        self.__model.zero_grad()


def plot_img(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
