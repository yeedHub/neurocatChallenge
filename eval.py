from collections import OrderedDict
from tqdm        import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.datasets   as datasets

from PyTorchModel import PyTorchNNMulticlassifier, PyTorchNNClassifierAnalyzer, PyTorchClassifierDataHandler

import external.resnet as resnet

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # taken from the resnet code
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

# In PyTorch the model, optimizer (e.g. Stocahstic Gradient Descent) 
# and loss (e.g. CrossEntropyLoss) are separate entities. 
# Also the model checkpoint has no information about the optimizer or loss function.
# This means that we have to look into the training code to see
# which optimizer and loss was used and instantiate them manually.
loss_function = torch.nn.CrossEntropyLoss()
optimizer     = torch.optim.SGD(resnet44model.parameters(), lr=0.1) # the learning rate used in the codebase was 0.1

# Our abstract model
loaderparams = {
  "batch_size": 10,
  "shuffle": True,
  "num_workers": 4
}
model = PyTorchNNMulticlassifier(resnet44model, optimizer, loss_function)
data  = PyTorchClassifierDataHandler([cifar10, cifar10_classes], loaderparams) 

stats1 = PyTorchNNClassifierAnalyzer(len(cifar10_classes))
stats2 = PyTorchNNClassifierAnalyzer(len(cifar10_classes))

for _ in tqdm(range(1)):
  X, Ytrue = data.getNextData(1)
  stats1.onePassAnalysis(model, X, Ytrue)
  stats2.multPassAnalysis(model, X, Ytrue)
  
print(stats1.accuracy, stats2.accuracy)
print(stats1.MacroF1, stats2.MacroF1)
print(stats1.WeightedF1, stats2.WeightedF1)
