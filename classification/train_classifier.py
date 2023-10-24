from datasets import Scan_DataModule
#from utils import Classifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import argparse
import os

import torch
import torchmetrics
import torch.nn.functional as F

import pytorch_lightning as pl

from models import VGG, ResNet, SimpleConvNet, ViTMelanoma

data_dir = '/home/scur0404/projects/MelanomaDetectionAndSegmentation/data_processed/classification/'

def get_config(args):
    config = {
        'train_data_dir' : os.path.join(data_dir, 'train'),
        'val_data_dir'   : os.path.join(data_dir, 'val'),
        'test_data_dir'  : os.path.join(data_dir, 'test'),
        'batch_size'     : args.batch_size,
        'optimizer_lr'   : args.optimizer_lr,
        'max_epochs'     : 50,
        'model_name'     : 'resnet18',
        'optimizer_name' : 'sgd',
        'bin'            : 'models/',
        'experiment_name': 'resnetprocessed',
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default: 32")
    parser.add_argument("--optimizer_lr", type=float, default=0.01, help="Learning rate for training. Default: 0.01")
    parser.add_argument("--freeze_until_layer", type=int, default=0, help="How many layers will be frozen. Default 0")

    args = parser.parse_args()

    config = get_config(args)

    wandb_logger = WandbLogger(project='moley-business', log_model=True)
    wandb_logger.experiment.config.update(config)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    data                = Scan_DataModule(config)
    classifier          = Classifier(config)
    logger              = wandb_logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/f1', mode = 'max')
    trainer             = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config['max_epochs'],
                                    logger=logger, callbacks=[checkpoint_callback],
                                    default_root_dir=config['bin'], deterministic=True,
                                    log_every_n_steps=1)
    trainer.fit(classifier, data)
    print('training completed')