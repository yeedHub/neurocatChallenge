from abc import ABCMeta, abstractmethod

class NNModelInterface(metaclass=ABCMeta):
  """
  The NNModelInterface generalizes neural network models of different
  frameworks.
  """
  @abstractmethod
  def __init__(self, model):
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