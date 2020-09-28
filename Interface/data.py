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
  