from abc import ABCMeta, abstractmethod

class DataHandlerInterface(metaclass=ABCMeta):
  """
  The DataHandlerInterface generalizes the data handling of different
  frameworks.
  """
  @abstractmethod
  def __init__(self, dataset, params):
    pass
  
  @abstractmethod
  def getNextData(self, count):
    """
    Depending on how the frameworks data pipeline is configured this
    function returns count number of data points or batches.
    """
    pass
  
  @abstractmethod
  def datasetLength(self):
    """
    Returns the number of samples in the dataset.
    """
    pass

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
  def forward(self, samples):
    """
    Forward step of the neural network model. samples can contain
    a single data point or multiple data points.
    """
    pass
  
  @abstractmethod
  def onePassAnalysis(self, samples):
    """
    One pass analyses group analyses that can be calculated with one pass over
    the given data points. This can for example be the calculation of accuracy.
    """
    pass
  
  @abstractmethod
  def multPassAnalysis(self, samples):
    """
    Multi-pass analyses group analyses that need more than one pass over the dataset
    to be calculated. For example in an image classification task one such analysis
    can use adversarial attacks (e.g. change one or multiple pixels of the
    images) and for each pass calculate one pass analyses and aggregate them in some
    form.
    """
    pass
  