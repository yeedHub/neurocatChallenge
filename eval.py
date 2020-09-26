import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.utils as utils

import matplotlib.pyplot as plt

mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tinydata    = datasets.ImageFolder("./data/imagenet/", transform=preprocess)
data_loader = torch.utils.data.DataLoader(tinydata, batch_size=3, shuffle=True, num_workers=4)

dataiter = iter(data_loader)
images, labels = dataiter.next()

def imshow(img):
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

output = mobilenet(images)#.unsqueeze(0))
normalized = torch.nn.functional.softmax(output, dim=0)
print(np.argmax(normalized.detach().numpy(), axis=1))

imshow(utils.make_grid(images))