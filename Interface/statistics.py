from abc import ABCMeta, abstractmethod
  
class ModelStatisticsInterface(metaclass=ABCMeta):
  """
  The ModelStatisticsInterface calculates different metrics for a given model.
  """
  @abstractmethod
  def __init__(self):
    pass
  
  @abstractmethod
  def onePassAnalysis(self, model, X):
    """
    One pass analyses group analyses that can be calculated with one pass over
    the given data points. This can for example be the calculation of accuracy.
    """
    pass
  
  @abstractmethod
  def multPassAnalysis(self, model, X):
    """
    Multi-pass analyses group analyses that need more than one pass over the dataset
    to be calculated. For example in an image classification task one such analysis
    can use adversarial attacks (e.g. change one or multiple pixels of the
    images) and for each pass calculate one pass analyses and aggregate them in some
    form.
    """
    pass