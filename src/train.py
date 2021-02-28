"""
Minimal training script from:
https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb
"""
import argparse
import os
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def get_args_parser():
    parser = argparse.ArgumentParser('Train', add_help=False)
    parser.add_argument('--num_epochs', default=50,)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', action='store_true')

    return parser


class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main(args: Namespace):
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)

    val_ds = MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    # Initialize a trainer
    logger = False
    checkpoint_callback = False
    callbacks = []
    if args.output_dir:
        logger = TensorBoardLogger(
            args.output_dir,
            version=1,
            default_hp_metric=False)
        checkpoint_callback = True
        callbacks.append(ModelCheckpoint(
            dirpath=args.output_dir,
            save_last=True,
            verbose=True))

    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = os.path.join(args.output_dir, 'last.ckpt')

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=args.output_dir,
        gpus=torch.cuda.device_count(),
        max_epochs=args.num_epochs,
        progress_bar_refresh_rate=20,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks)

    # Train the model
    trainer.fit(
        mnist_model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train', parents=[get_args_parser()])
    args = parser.parse_args()

    checkpoint_file = os.path.join(args.output_dir, 'last.ckpt')
    if os.path.exists(checkpoint_file):
        args.resume = True

    main(args)
