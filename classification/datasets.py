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

      # setting label from image name
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