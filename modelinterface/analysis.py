from abc import ABCMeta, abstractmethod
  
class ModelAnalyzerInterface(metaclass=ABCMeta):
  """
  The ModelStatisticsInterface calculates different metrics for a given model.
  """
  @abstractmethod
  def __init__(self):
    pass
  
  @abstractmethod
  def functionality_analysis(self, model, X, params):
    """
    Calculate all functionality related metrics.
    """
    pass
  
  @abstractmethod
  def robustness_analysis(self, model, X, params):
    """
    Calculate all robustness related metrics.
    """
    pass