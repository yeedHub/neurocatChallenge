from collections import OrderedDict
from tqdm        import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.datasets   as datasets

from PyTorchModel import PyTorchNNMulticlassifier, PyTorchNNClassifierStatistics, PyTorchClassifierDataHandler

import external.resnet as resnet

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10 = datasets.CIFAR10("./data/CIFAR10/", train=False, transform=preprocess, download=False)
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
resnet44model = resnet.resnet44()
resnet44model.load_state_dict(new_state_dict)

# Our abstract model
loaderparams = {
  "batch_size": 5,
  "shuffle": True,
  "num_workers": 4
}
model = PyTorchNNMulticlassifier(resnet44model)
data  = PyTorchClassifierDataHandler([cifar10, cifar10_classes], loaderparams) 
stats = PyTorchNNClassifierStatistics(len(cifar10_classes))

for _ in tqdm(range(20)):
  X, Ytrue = data.getNextData(1)
  stats.onePassAnalysis(model, X, Ytrue)
  
print(stats.accuracy())
print(stats.MacroF1(), stats.WeightedF1())