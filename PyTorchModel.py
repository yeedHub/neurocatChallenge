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
  def __init__(self, model, optimizer, loss_function):
    self.__model         = model
    self.__optimizer     = optimizer
    self.__loss_function = loss_function
  
  def forward(self, x):
    """
    x has to be 4 dimensional: batch X RGB X H x W 
    """
    self.__model.eval()
    return self.__model(x)

  def interpret_output(self, output):
    output        = torch.nn.functional.softmax(output, dim=0)
    probabilities = np.max(output.detach().numpy(), axis=0)
    probargmax    = np.argmax(output.detach().numpy(), axis=1)
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
    ytrue           = torch.from_numpy(ytrue)
    
    output = self.forward(x)
      
    self.zero_grad()
    self.backward(output, ytrue)
    
    return x.grad.data
  
  def zero_grad(self):
    self.__model.zero_grad()

class PyTorchNNClassifierAnalyzer(ModelAnalyzerInterface):  
  def __init__(self, num_class):
    self.num_class = num_class                     # number of different classes
    
    self.total_samples       = 0                   # number of all samples
    self.total_class_samples = np.zeros(num_class) # number of each class in data 
    
    self.cnf_matrix = np.zeros((num_class, num_class))
    self.TrueP      = np.zeros(num_class)
    self.TrueN      = np.zeros(num_class)
    self.FalseP     = np.zeros(num_class)
    self.FalseN     = np.zeros(num_class)
    
    self.accuracy   = 0
    self.MacroF1    = 0
    self.WeightedF1 = 0
  
  def functionality_analysis(self, model, X, Ytrue, params=None):
    for i, x in enumerate(X):
      interpreted_output = model.interpret_output(model.forward(x))
      
      self.total_samples += len(interpreted_output[0])
      for s in Ytrue[i]:
        self.total_class_samples[s] += 1
      
      self.calculate_confusion_matrix(interpreted_output[0], Ytrue[i])
    
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
    
    denominator      = np.sum(self.TrueN) + np.sum(self.FalseP)
    self.specificity = np.sum(self.TrueN) / denominator if denominator != 0 else 1
    
    denominator    = np.sum(self.TrueP) + np.sum(self.FalseP)
    self.precision = np.sum(self.TrueP) / denominator if denominator != 0 else 1
    
    denominator = np.sum(self.TrueP) + np.sum(self.FalseN)
    self.recall = np.sum(self.TrueP) / denominator if denominator != 0 else 1
    
    denominator = self.precision + self.recall
    self.F1     = 2 * ( self.precision * self.recall ) / denominator if denominator != 0 else 1
    
    self.WeightedF1   = np.sum(self.F1 * self.total_class_samples) / self.total_samples
    self.bal_accuracy = np.sum(self.recall + self.specificity) / 2.0
    self.accuracy     = (np.sum(self.TrueP) + np.sum(self.TrueN)) / (np.sum(self.TrueP) + np.sum(self.TrueN) + np.sum(self.FalseP) + np.sum(self.FalseN))
  
  def robustness_analysis(self, model, X, Ytrue, params=None):
    result = []
    for i, x in enumerate(X):
      
      gradient  = model.gradient_for(x, Ytrue[i])
      perturbed = None
      if params != None and "fsgm_eps" in params:
        perturbed = self.fgsm_attack(params["fsgm_eps"], x, gradient)
      else:
        perturbed = self.fgsm_attack(0.007, x, gradient)        
      
      # If the prediction of original data is wrong, don't include them
      # (otherwise we run the risk of skewing our attack result)
      interpreted_output = model.interpret_output(model.forward(x))
      mask               = interpreted_output[0] != Ytrue[i]
      perturbed[mask]    = x[mask]
      
      result.append(perturbed)  
      # images = utils.make_grid(torch.cat((x, perturbed), 0))
      # plot_img(images)
    
    self.functionality_analysis(model, result, Ytrue)
    # images = utils.make_grid(torch.cat(result, 0))
    # plot_img(images)
    
  def fgsm_attack(self, epsilon, x, grad):
      perturbed = x + epsilon*grad.sign()
      perturbed = torch.clamp(perturbed, np.min(x.detach().numpy()), np.max(x.detach().numpy()))
      
      return perturbed
  
def plot_img(img):
  npimg = img.detach().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()
  
