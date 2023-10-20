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

from models import VGG, ResNet, SimpleConvNet, ViTMelanoma
from train_classifier import get_config

config = get_config()

models     = {'vgg16_smaller' : VGG,
              'resnet18'      : ResNet,
              'custom_convnet': SimpleConvNet,
              'ViTMelanoma'   : ViTMelanoma }


optimizers = {'adam'          : torch.optim.Adam,
              'sgd'           : torch.optim.SGD }

metrics    = {'acc'           : torchmetrics.Accuracy(task='binary').to('cuda'),
              'f1'            : torchmetrics.F1Score(task='binary').to('cuda'),
              'precision'     : torchmetrics.Precision(task='binary').to('cuda'),
              'recall'        : torchmetrics.Recall(task='binary').to('cuda')}

class Classifier(pl.LightningModule):
  def __init__(self, *args):
    super().__init__()

    # defining model
    self.model_name = config['model_name']
    assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
    if config['model_name'] == 'ViTMelanoma':
      self.model      = models[self.model_name](config['freeze_until_layer']).to('cuda')
    else:
      self.model      = models[self.model_name]().to('cuda')

    # assigning optimizer values
    self.optimizer_name = config['optimizer_name']
    self.lr             = config['optimizer_lr']

  def step(self, batch, nn_set):
    X, y   = batch
    X, y   = X.float().to('cuda'), y.to('cuda')

    y_hat  = self.model(X).squeeze(1)
    y_prob = torch.sigmoid(y_hat)

    loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
    self.log(f'{nn_set}/loss', loss, on_step=False, on_epoch=True)

    for metric_name, metric_fn in metrics.items():
      score = metric_fn(y_prob, y)
      self.log(f'{nn_set}/{metric_name}', score, on_step=False, on_epoch=True)

    return loss

  def training_step(self, batch, batch_idx):
    return {"loss": self.step(batch, "train")}

  def validation_step(self, batch, batch_idx):
    return {"val_loss": self.step(batch, "val")}

  def test_step(self, batch, batch_idx):
    return {"test_loss": self.step(batch, "test")}

  def forward(self, X):
    return self.model(X)

  def configure_optimizers(self):
    assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
    return optimizers[self.optimizer_name](self.parameters(), lr = self.lr)

