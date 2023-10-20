import glob
import os
from train_classifier import Classifier, get_config
from datasets import Scan_DataModule
import pytorch_lightning as pl
import argparse
import wandb

def load_checkpoint(path_to_ckpt):
    api = wandb.Api()
    artifact = api.artifact(path_to_ckpt)
    artifact_dir = artifact.download()
    return artifact_dir + '/model.ckpt'

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
model = Classifier(config).load_from_checkpoint(PATH)
model.eval()

# make test dataloader
test_data = Scan_DataModule(config)

# test model
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config['max_epochs'],
                    #logger=logger, callbacks=[checkpoint_callback],
                    default_root_dir=config['bin'], deterministic=True,
                    log_every_n_steps=1)
trainer.test(model, dataloaders=test_data, verbose=True)