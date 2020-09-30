from abc import ABCMeta, abstractmethod

class NNModelInterface(metaclass=ABCMeta):
  """
  The NNModelInterface generalizes neural network models of different
  frameworks.
  """
  @abstractmethod
  def __init__(self, model, optimizer, loss):
    """
    The constructor takes a reference to the model.
    The exact form of these references depend on the used framework.
    """
    pass
  
  @abstractmethod
  def forward(self, x):
    """
    Forward step of the neural network model. x is supposed to be a
    single data point (or batch).
    """
    pass
  
  @abstractmethod
  def interpret_output(self, output):
    """
    Interpret the output of self.forward(x). This can be useful e.g. when we build a
    multi-class classifier and our models output needs to be normalized to probabilities.
    """
    pass
  
  @abstractmethod
  def backward(self, output):
    """
    Backward step of the neural network model. output has to be the direct model output, 
    before calling self.interpret_output(output).
    """
    pass
  
  @abstractmethod
  def apply_gradients(self):
    """
    Applies the gradients to the weights of the model. This function should be called
    after self.backward(output).
    """
    pass
  
  @abstractmethod
  def zero_grad(self):
    """
    Set all gradients to zero.
    """
    pass
  
  @abstractmethod
  def gradient_for(self, x):
    """
    Returns the gradient for x. x is assumed to be a single data point (or batch).
    """
    pass
  