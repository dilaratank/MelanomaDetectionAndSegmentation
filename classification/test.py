'''
Code copied from Notebook provided by pracitcal.
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

from models import VGG, ResNet, SimpleConvNet, ViTMelanoma

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

data_dir = '/home/scur0404/projects/MelanomaDetectionAndSegmentation/data_processed/classification/'

# we are aware of the code duplication with train_classifier.py, but since the code we got from the notebook
# is not formatted in a way for us to be able to use it as python files, we fixed it like this.
# With more time we could have prevented this code duplication. 

def get_config(args):
    config = {
        'train_data_dir' : os.path.join(data_dir, 'train'),
        'val_data_dir'   : os.path.join(data_dir, 'val'),
        'test_data_dir'  : os.path.join(data_dir, 'test'),
        'batch_size'     : args.batch_size,
        'optimizer_lr'   : args.optimizer_lr,
        'max_epochs'     : 10,
        'model_name'     : 'resnet18',
        'optimizer_name' : 'sgd',
        'bin'            : 'models/',
        'experiment_name': 'resnet18',
        'num_workers'    :  8,
        'freeze_until_layer': args.freeze_until_layer
    }
    return config

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

def load_checkpoint(path_to_ckpt):
    api = wandb.Api()
    artifact = api.artifact(path_to_ckpt)
    artifact_dir = artifact.download()
    return artifact_dir + '/model.ckpt'

class Scan_Dataset(Dataset):
    def __init__(self, data_dir, transform=False):
      self.transform = transform
      self.data_list = sorted(glob.glob(os.path.join(data_dir,'img*.nii.gz')))

    def __len__(self):
      """defines the size of the dataset (equal to the length of the data_list)"""
      return len(self.data_list)

    def __getitem__(self, idx):
      """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""
      if torch.is_tensor(idx):
        idx = idx.tolist()

      # loading image
      image_name = self.data_list[idx]
      image = nib.load(image_name).get_fdata()
      #image = np.transpose(image, (2, 0, 1))

      label = int(image_name.split('.')[0][-1])
      label = torch.tensor(label)

      # apply transforms
      if self.transform:
        image = self.transform(image)

      return image, label
    

def visualize_img(data_dir):
  #@title Visualizing Images { run: 'auto'}
    nn_set = 'test' #@param ['train', 'val', 'test']
    index  = 0 #@param {type:'slider', min:0, max:355, step:5}

    dataset  = Scan_Dataset(os.path.join(data_dir, nn_set))

    n_images_display = 5
    fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
    for i in range(n_images_display):
        image, label = dataset[index+i]
        ax[i].imshow(np.uint8(image))
        #ax[i].imshow(np.uint8(np.transpose(image, (1, 2, 0))));
        ax[i].set_title(f'Cancer : {"Yes" if label else "No"}')
        ax[i].axis('off')
    plt.suptitle(f'Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0][0].shape}]');

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
  
class Scan_DataModule(pl.LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']

    self.train_transforms = transforms.Compose([Random_Rotate(0.1), transforms.ToTensor()])
    self.val_transforms  = transforms.Compose([transforms.ToTensor()])

  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset(self.val_data_dir  , transform = self.val_transforms)
    self.test_dataset = Scan_Dataset(self.test_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    loader = DataLoader(self.train_dataset, batch_size = self.batch_size)
    return DataLoader(self.train_dataset, batch_size = self.batch_size)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=8)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default: 32")
    parser.add_argument("--optimizer_lr", type=float, default=0.01, help="Learning rate for training. Default: 0.01")
    parser.add_argument("--freeze_until_layer", type=int, default=0, help="How many layers will be frozen. Default 0")
    parser.add_argument("--checkpoint_folder_path", type=str, help="Path to model checkpoint folder")

args = parser.parse_args()

config = get_config(args)

# load best model
PATH = load_checkpoint(args.checkpoint_folder_path)
model = Classifier.load_from_checkpoint(PATH)
model.eval()

# make test dataloader
test_data = Scan_DataModule(config)

# test model
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config['max_epochs'],
                    #logger=logger, callbacks=[checkpoint_callback],
                    default_root_dir=config['bin'], deterministic=True,
                    log_every_n_steps=1)

print(trainer.test(model, dataloaders=test_data, verbose=True))