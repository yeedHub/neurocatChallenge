from Interface.analysis import ModelAnalyzerInterface
from Interface.nn       import NNModelInterface
from Interface.data     import DataHandlerInterface

import torch
import torchvision.utils as utils
import numpy             as np
import matplotlib.pyplot as plt

from random import seed
from random import randint
from random import sample

# from sklearn.metrics import confusion_matrix

seed(1)

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
      
    return (samples, np.array(labels))
  
  def datasetLength(self):
    return self.__dataset.__len__()

class PyTorchNNMulticlassifier(NNModelInterface):
  def __init__(self, model):
    self.__model = model
  
  def forward(self, x):
    """
    x has to be 4 dimensional: batch X RGB X H x W 
    """
    self.__model.eval()
    
    output = self.__model(x)
    output = torch.nn.functional.softmax(output, dim=0)
      
    outargmax = np.argmax(output.detach().numpy(), axis=1)
      
    return outargmax

class PyTorchNNClassifierAnalyzer(ModelAnalyzerInterface):  
  def __init__(self, num_class):
    self.num_class = num_class                       # number of different classes
    
    self.total_samples       = 0                     # number of all samples
    self.total_class_samples = np.zeros(num_class) # number of each class in data 
    
    self.cnf_matrix = np.zeros((num_class, num_class))
    self.TrueP      = np.zeros(num_class)
    self.TrueN      = np.zeros(num_class)
    self.FalseP     = np.zeros(num_class)
    self.FalseN     = np.zeros(num_class)
    
    self.accuracy = 0
  
  def onePassAnalysis(self, model, X, Ytrue):
    for i, x in enumerate(X):
      output = model.forward(x)
      
      self.total_samples += len(output)
      for s in Ytrue[i]:
        self.total_class_samples[s] += 1
      
      self.calculate_confusion_matrix(output, Ytrue[i])
    
    self.calculate_cnf_derivations()
  
  def calculate_confusion_matrix(self, output, ytrue):
    # self.cnf_matrix += confusion_matrix(ytrue, output, labels=range(self.num_class))
    for i, pred in enumerate(output):
      if pred == ytrue[i]:
        self.cnf_matrix[pred, pred] += 1
      else:
        self.cnf_matrix[ytrue[i], pred] += 1
    
  def calculate_cnf_derivations(self):
    self.TrueP  = np.diag(self.cnf_matrix)    
    self.FalseP = self.cnf_matrix.sum(axis=0) - self.TrueP
    self.FalseN = self.cnf_matrix.sum(axis=1) - self.TrueP
    self.TrueN  = self.cnf_matrix.sum() - (self.TrueP + self.FalseN + self.FalseP)
    
    self.accuracy = np.sum(self.TrueP + self.TrueN) / np.sum(self.TrueP + self.TrueN + self.FalseP + self.FalseN)
    
    # Setting those elements in the denominator that are 0 to 1 results in a valid division
    # but does not change the result - since TrueP is also in the numerator
    denominator = (self.TrueP + self.FalseP)
    denominator[denominator == 0] = 1
    self.precision = self.TrueP / denominator
    
    # Setting those elements in the denominator that are 0 to 1 results in a valid division
    # but does not change the result - since TrueP is also in the numerator
    denominator = (self.TrueP + self.FalseN)
    denominator[denominator == 0] = 1
    self.recall = self.TrueP / denominator
    
    # Setting those elements in the denominator that are 0 to 1 results in a valid division
    # but does not change the result - since self.precision() * self.recall() is definitly zero in the numerator
    # if both are 0.
    denominator = (self.precision + self.recall)
    denominator[denominator == 0] = 1
    self.F1 = 2 * ( self.precision * self.recall ) / denominator
    
    self.MacroF1    = np.sum(self.F1) / self.num_class
    self.WeightedF1 = np.sum(self.F1 * self.total_class_samples) / self.total_samples
  
  # TODO: clean this up somehow?? This function definitly needs testing.
  def multPassAnalysis(self, model, X, Ytrue):
    percentage = range(0, 11)[randint(0, 10)] / 100.0
    for i, x in enumerate(X):
      for j in range(x.size()[0]):
        sample_numpy           = x[j].numpy()
        sample_size            = x[j].size()
        num_features_to_change = np.prod(sample_size) * percentage
        
        # Create num_features_to_change indeces that fit the sample.size() (random in all dimensions)
        idx = [tuple(randint(0, sample_size[k]-1) for k in range(0, len(sample_size))) for _ in  range(0, int(num_features_to_change))]
        
        for l in range(0, len(idx)):
          X[i][j][idx[l]] = randint(int(np.min(sample_numpy)), int(np.max(sample_numpy)) * 100) / 100.0

      self.onePassAnalysis(model, X, Ytrue)