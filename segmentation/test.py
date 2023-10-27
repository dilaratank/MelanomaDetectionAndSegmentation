'''
Code copied from Notebook provided by pracitcal.
We replaces the tensorboard logger with the wandb logger.
'''

import glob
import os
import pytorch_lightning as pl
import argparse
import wandb

import torch
import torchmetrics
import torch.nn.functional as F

import pytorch_lightning as pl

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
from models import UNet

data_dir = '/home/scur0404/projects/MelanomaDetectionAndSegmentation/data/segmentation/'

test_data_dir = os.path.join(data_dir, 'val')

# we are aware of the code duplication with train_classifier.py, but since the code we got from the notebook
# is not formatted in a way for us to be able to use it as python files, we fixed it like this.
# With more time we could have prevented this code duplication. 

def load_checkpoint(path_to_ckpt):
    api = wandb.Api()
    artifact = api.artifact(path_to_ckpt)
    artifact_dir = artifact.download()
    return artifact_dir + '/model.ckpt'

def get_config(args):
    config_segm = {
        'train_data_dir' : os.path.join(data_dir, 'train'),
        'val_data_dir'   : os.path.join(data_dir, 'val'),
        'test_data_dir'  : os.path.join(data_dir, 'test'),
        'batch_size'     : args.batch_size,
        'optimizer_lr'   : args.optimizer_lr,
        'loss_pos_weight': args.loss_pos_weight,  # Assigning a weight for positive examples is one solution to class imbalance.
        'max_epochs'     : 100,
        'model_name'     : 'unet',
        'optimizer_name' : 'adam',
        'bin'            : 'segm_models/',
        'experiment_name': 'adam'
    }
    return config_segm

models     = {'unet'      : UNet}

optimizers = {'adam'      : torch.optim.Adam,
              'sgd'       : torch.optim.SGD }

metrics    = {'acc'       : torchmetrics.Accuracy(task='binary').to('cuda'),
              'f1'        : torchmetrics.F1Score(task='binary').to('cuda'),
              'precision' : torchmetrics.Precision(task='binary').to('cuda'),
              'recall'    : torchmetrics.Recall(task='binary').to('cuda'),
              'dice'      : torchmetrics.Dice().to('cuda')}

class Segmenter(pl.LightningModule):
  def __init__(self, *args):
    super().__init__()

    # defining model
    self.model_name = config_segm['model_name']
    assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
    self.model      = models[self.model_name]().to('cuda')

    # assigning optimizer values
    self.optimizer_name = config_segm['optimizer_name']
    self.lr             = config_segm['optimizer_lr']

  def step(self, batch, nn_set):
    X, y = batch['image'], batch['mask']
    X, y   = X.float().to('cuda'), y.to('cuda').float()
    y_hat  = self.model(X)
    y_prob = torch.sigmoid(y_hat)

    pos_weight = torch.tensor([config_segm['loss_pos_weight']]).float().to('cuda')
    #loss = F.binary_cross_entropy_with_logits(y, y_prob, pos_weight=pos_weight)
    loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), pos_weight=pos_weight)
    del X, y_hat, batch
    
    self.log(f"{nn_set}/loss", loss, on_step=False, on_epoch=True)

    for i, (metric_name, metric_fn) in enumerate(metrics.items()):
      score = metric_fn(y_prob, y.int())
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

class Scan_Dataset_Segm(Dataset):
  def __init__(self, data_dir, transform=False):
    self.transform = transform
    self.img_list = sorted(glob.glob(os.path.join(data_dir,'img*.nii.gz')))
    self.msk_list = sorted(glob.glob(os.path.join(data_dir,'msk*.nii.gz')))

  def __len__(self):
    """defines the size of the dataset (equal to the length of the data_list)"""
    return len(self.img_list)

  def __getitem__(self, idx):
    """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""

    if torch.is_tensor(idx):
      idx = idx.tolist()

    # loading image
    image_name = self.img_list[idx]
    image = nib.load(image_name).get_fdata()

    # loading mask
    mask_name = self.msk_list[idx]
    mask = nib.load(mask_name).get_fdata()
    mask = np.expand_dims(mask, axis=2)

    # make sample
    sample = {'image': image, 'mask': mask}

    # apply transforms
    if self.transform:
      sample = self.transform(sample)

    return sample
  
class Random_Rotate_Seg(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      image = rotate(image, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
      mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    return {'image': image.copy(), 'mask': mask.copy()}

class ToTensor_Seg(object):
  """applies ToTensor for dict input"""
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    return {'image': image.clone(), 'mask': mask.clone()}
  
class Scan_DataModule_Segm_Test(pl.LightningDataModule):
    def __init__(self, test_data_dir, batch_size=1):
      super().__init__()
      self.test_data_dir    = test_data_dir
      self.batch_size       = batch_size

      self.test_transforms = transforms.Compose([ToTensor_Seg()])

    def setup(self, stage=None):
        self.test_dataset  = Scan_Dataset_Segm(self.test_data_dir , transform = self.test_transforms)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size)

class Random_Rotate(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      sample = rotate(sample, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    return sample.copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default: 32")
    parser.add_argument("--optimizer_lr", type=float, default=0.001, help="Learning rate for training. Default: 0.01")
    parser.add_argument("--loss_pos_weight", type=int, default=2, help=". Default 2")
    parser.add_argument("--checkpoint_folder_path", type=str, help="Path to model checkpoint folder")

args = parser.parse_args()

config_segm = get_config(args)

# load best model
PATH = load_checkpoint(args.checkpoint_folder_path)
model = Segmenter.load_from_checkpoint(PATH)
model.eval()

# make test dataloader
test_data = Scan_DataModule_Segm_Test(test_data_dir)

# test model
trainer = pl.Trainer(devices=1, max_epochs=config_segm['max_epochs'],
                    #logger=logger, callbacks=[checkpoint_callback],
                    default_root_dir=config_segm['bin'], deterministic=True,
                    log_every_n_steps=1)
print(trainer.test(model, dataloaders=test_data, verbose=True))