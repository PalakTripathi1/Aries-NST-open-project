from PIL import Image
import matplotlib.pyplot as plots
import numpy as py

import torch
import torch.optim as op
from torchvision import transforms, models
# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vg = models.vgg19(pretrained=True).features
