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

from models import UNet
from datasets import Scan_DataModule_Segm

data_dir = '../data/segmentation/'

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
    del X, y_hat, batch

    pos_weight = torch.tensor([config_segm['loss_pos_weight']]).float().to('cuda')
    loss = F.binary_cross_entropy_with_logits(y, y_prob, pos_weight=pos_weight)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default: 32")
    parser.add_argument("--optimizer_lr", type=float, default=0.1, help="Learning rate for training. Default: 0.01")
    parser.add_argument("--loss_pos_weight", type=int, default=2, help=". Default 2")

    args = parser.parse_args()

    config_segm = get_config(args)

    wandb_logger = WandbLogger(project='moley-business', log_model=True)
    wandb_logger.experiment.config.update(config_segm)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    data                = Scan_DataModule_Segm(config_segm)
    segmenter           = Segmenter(config_segm)
    logger              = wandb_logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/f1')
    trainer             = pl.Trainer(devices=1, accelerator='gpu', max_epochs=config_segm['max_epochs'],
                                    logger=logger, callbacks=[checkpoint_callback],
                                    default_root_dir=config_segm['bin'], deterministic=True,
                                    log_every_n_steps=1)
    trainer.fit(segmenter, data)