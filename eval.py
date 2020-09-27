from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.utils as utils

from PyTorchModel import PyTorchNNModel 
from PyTorchModel import PyTorchClassifierDataHandler
from PyTorchModel import visualizeImages

import external.resnet as resnet

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

cifar10 = datasets.CIFAR10("./data/CIFAR10/", transform=preprocess, download=False)
cifar10_classes = ["airplane",
                   "automobile",
                   "bird",
                   "cat",
                   "deer",
                   "dog",
                   "frog",
                   "horse",
                   "ship",
                   "truck"]

# Load the pre-trained checkpoint for the ResNet44 model (trained on CIFAR10)
checkpoint = torch.load("./external/pretrained_models/resnet44-014dd654.th", map_location=torch.device('cpu'))

# The model was saved wrapped in torch.nn.DataParallel, therefore we have to strip the "module." prefix
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
  name = k[7:] # remove "module."
  new_state_dict[name] = v
  
# Load in the state_dict of the pre-trained model
resnet44model = resnet.resnet44().load_state_dict(new_state_dict)

# Our abstract model
loaderparams = {
  "batch_size": 3,
  "shuffle": True,
  "num_workers": 4
}
model = PyTorchNNModel(resnet44model)
data  = PyTorchClassifierDataHandler([cifar10, cifar10_classes], loaderparams) 
x, ytrue = data.getNextData(1)
visualizeImages(x)
print(data.datasetLength())