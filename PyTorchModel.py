from ModelInterface import NNModelInterface
from ModelInterface import DataHandlerInterface

import torch
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt

class PyTorchClassifierDataHandler(DataHandlerInterface):
  def __init__(self, dataset, params):
    self.__dataset    = dataset[0] # contains x and y_true
    self.__charlabels = dataset[1] # human readable labels
    self.configureDataLoader(params)
    self.__dataiter = iter(self.__loader)
    
  def configureDataLoader(self, params):
    self.__loader = torch.utils.data.DataLoader(self.__dataset, **params)
    
  def getNextData(self, count):
    samples = []
    labels  = []
    
    for _ in range(count):
      datapoint = self.__dataiter.next()
      samples.append(datapoint[0])
      labels.append(datapoint[1])
      
    return (samples, labels)
  
  def datasetLength(self):
    return self.__dataset.__len__()

class PyTorchNNModel(NNModelInterface):
  def __init__(self, model):
    self.__model = model
  
  def forward(self, samples):
    self.__model.eval()
    for sample in samples:
      output = self.__model(sample)
      # output = torch.nn.functional.softmax(output, dim=0)
      # print(np.argmax(output.detach().numpy(), axis=1))
      print(output)
  
  def onePassAnalysis(self, samples):
    pass
  
  def multPassAnalysis(self, samples):
    pass
    

# Small helper function to plot images
def visualizeImages(imgs):
  npimg = utils.make_grid(imgs).numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()