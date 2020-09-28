from Interface.statistics import ModelStatisticsInterface
from Interface.nn         import NNModelInterface
from Interface.data       import DataHandlerInterface

import torch
import torchvision.utils as utils
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class PyTorchClassifierDataHandler(DataHandlerInterface):
  def __init__(self, dataset, params):
    self.__dataset    = dataset[0] # contains x and y_true
    self.__charlabels = dataset[1] # human readable labels
    self.__loader     = torch.utils.data.DataLoader(self.__dataset, **params)
    self.__dataiter   = iter(self.__loader)
  
  # TODO: does not handle out of data request  
  def getNextData(self, count):
    samples = []
    labels  = []
    
    for _ in range(count):
      x, y = self.__dataiter.next()
      
      if (isinstance(y, torch.Tensor)):
        y = y.detach().numpy()
        
      samples.append(x)
      labels.append(y)
      
    return (samples, labels)
  
  def datasetLength(self):
    return self.__dataset.__len__()

class PyTorchNNMulticlassifier(NNModelInterface):
  def __init__(self, model):
    self.__model = model
  
  def forward(self, x):
    self.__model.eval()
    
    output = self.__model(x)
    output = torch.nn.functional.softmax(output, dim=0)
      
    outargmax = np.argmax(output.detach().numpy(), axis=1)
      
    return outargmax

class PyTorchNNClassifierStatistics(ModelStatisticsInterface):  
  def __init__(self, class_count):
    self.class_count = class_count                   # number of different classes
    
    self.total_samples       = 0                     # number of all samples
    self.total_class_samples = np.zeros(class_count) # number of each class in data 
    
    self.cnf_matrix = np.zeros((class_count, class_count))
    self.TrueP      = np.zeros(class_count)
    self.TrueN      = np.zeros(class_count)
    self.FalseP     = np.zeros(class_count)
    self.FalseN     = np.zeros(class_count)
  
  def onePassAnalysis(self, model, X, Ytrue):
    for i, x in enumerate(X):
      output = model.forward(x)
      
      self.cnf_matrix += confusion_matrix(Ytrue[i], output, labels=range(self.class_count))
      
      self.total_samples                 += len(output)
      self.total_class_samples[Ytrue[i]] += 1
      
    self.TrueP  = np.diag(self.cnf_matrix)    
    self.FalseP = self.cnf_matrix.sum(axis=0) - self.TrueP
    self.FalseN = self.cnf_matrix.sum(axis=1) - self.TrueP
    self.TrueN  = self.cnf_matrix.sum() - (self.TrueP + self.FalseN + self.FalseP)
  
  def multPassAnalysis(self, X):
    pass
  
  def accuracy(self):
    return np.sum(self.TrueP + self.TrueN) / np.sum(self.TrueP + self.TrueN + self.FalseP + self.FalseN)
  
  def cnf_matrix(self):
    return {"matrix": self.cnf_matrix, "TP": self.TrueP, "FP": self.FalseP, "TN": self.TrueN, "FN": self.FalseN}
  
  def precision(self):
    return self.TrueP / (self.TrueP + self.FalseP)
  
  def recall(self):
    return self.TrueP / (self.TrueP + self.FalseN)
  
  def F1(self):
    return 2 * ( self.precision() * self.recall() ) / ( self.precision() + self.recall() )
  
  def MacroF1(self):
    return np.sum(self.F1()) / self.class_count
  
  def WeightedF1(self):
    return np.sum(self.F1() * self.total_class_samples)