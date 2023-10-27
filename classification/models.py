'''
Code copied from Notebook provided by pracitcal.
'''
import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

import torch
import torchvision
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class SimpleConvNet(pl.LightningModule):

  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
        # conv block 1
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        # conv block 2
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        # conv block 3
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2))


    self.classifier = nn.Sequential(
        # linear layers
        nn.AdaptiveAvgPool2d(output_size=(4,4)),
        nn.Flatten(),
        nn.Linear(in_features=4*4*64, out_features=120),
        nn.ReLU(),
        nn.Linear(in_features=120, out_features=1)
    )

  def forward(self, x):
    x = self.layers(x)
    x = self.classifier(x)
    return x
  
class ResNet(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.model = torchvision.models.resnet18(pretrained=True)
    # We change the input and output layers to make the model compatible to our data
    self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
    self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

  def forward(self, x):
    x = self.model(x)
    return x
  
class VGG(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.model = torchvision.models.vgg16(pretrained=True)
    # We change the input and output layers to make the model compatible to our data
    self.model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.model.avgpool = nn.Identity()
    self.model.classifier = nn.Sequential(nn.Linear(in_features=4*4*512, out_features=512, bias=True),
                              nn.ReLU(inplace=True),
                              nn.Dropout(p=0.5, inplace=False),
                              nn.Linear(in_features=512, out_features=1, bias=True))
  def forward(self, x):
    x = self.model(x)
    return x
  
class ViTMelanoma(pl.LightningModule):
  def __init__(self, freeze_until_layer):
    super().__init__()
    extractor = AutoFeatureExtractor.from_pretrained("UnipaPolitoUnimore/vit-large-patch32-384-melanoma")
    model = AutoModelForImageClassification.from_pretrained("UnipaPolitoUnimore/vit-large-patch32-384-melanoma")

    self.model = model
    self.model.classifier = nn.Linear(in_features=1024, out_features=1, bias=True)

    self.freeze_until_layer = freeze_until_layer

    for name, param in list(model.named_parameters()):
      if str(self.freeze_until_layer) in name:
        break
      param.requires_grad = False

  def forward(self, x):
    #print('shape before interpolate', x.shape)
    x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
    #print('shape before model', x.shape)
    x = self.model(x)
    return x['logits']