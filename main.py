import os
from glob import glob
import argparse

import numpy as np
import cv2

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchmetrics
import matplotlib.pyplot as plt

from unet import Unet


torch.set_float32_matmul_precision("high")


def parse_options():
    parser = argparse.ArgumentParser('Semantic Segmentation using UNet')

    parser.add_argument('mode', type=str, choices=['train', 'predict'],
                        help='Mode to run the script in')

    parser.add_argument('--exp-path', type=str, default='exp',
                        help='Path to store experiment results under experiments/')

    parser.add_argument('--checkpoint', type=str,
                        help='Path to the model checkpoint')

    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the dataset')

    parser.add_argument('--output-path', type=str, default='output',
                        help='Path to store predictions')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')

    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs for training')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers for data loading')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    parser.add_argument('--seed', type=int, default=7829347,
                        help='Seed for reproducibility')

    args = parser.parse_args()
    args.exp_path = os.path.join('experiments', args.exp_path)

    return args


class ImageSeg(Dataset):
    DEFAULT_TRANSFORMS = A.Compose([
        A.ToFloat(max_value=255),
        ToTensorV2(),
    ])

    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.images = sorted(glob(os.path.join(root, split, 'images', '*.png')))
        self.masks = sorted(glob(os.path.join(root, split, 'masks', '*.png')))

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx])[:, :, 0]

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        augmented = self.DEFAULT_TRANSFORMS(image=image, mask=mask)

        return {
            'image': augmented['image'],
            'mask': augmented['mask'],
            'filename': os.path.basename(self.images[idx])
        }

    def __len__(self):
        return len(self.images)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        # flatten label and prediction tensors

        num_classes = inputs.shape[1]
        inputs = inputs.flatten()

        targets_onehot = F.one_hot(targets.long(), num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2)
        targets = targets_onehot.flatten()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def vis(model, dataset, device):
    # randomly select 10 samples and visualize images/masks/predictions
    # in a 10x3 grid
    indices = np.random.choice(len(dataset), 10)

    samples = [dataset[i] for i in indices]

    images = torch.stack([sample['image'] for sample in samples]).to(device)
    masks = torch.stack([sample['mask'] for sample in samples]).to(device)
    outputs = model(images).argmax(1)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = outputs.cpu().numpy()

    cmap = np.array(
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8)

    masks = cmap[masks]
    outputs = cmap[outputs]

    fig, axes = plt.subplots(10, 3, figsize=(10, 30), dpi=100)
    for i in range(10):
        axes[i, 0].imshow(images[i].transpose(1, 2, 0))
        axes[i, 1].imshow(masks[i].squeeze())
        axes[i, 2].imshow(outputs[i].squeeze())
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')

    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].set_title('Prediction')

    axes[0, 1].legend(handles=[
        plt.Line2D([0], [0], color='black', lw=4, label='Class 0'),
        plt.Line2D([0], [0], color='red', lw=4, label='Class 1'),
        plt.Line2D([0], [0], color='green', lw=4, label='Class 2'),
        plt.Line2D([0], [0], color='blue', lw=4, label='Class 3'),
    ], loc='upper right')

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.tight_layout()
    return fig


class UNet(L.LightningModule):
    def __init__(self, class_num=4, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = Unet(backbone_name='resnet50',
                          classes=class_num)
        self.criterion = DiceLoss()

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.train_cm = torchmetrics.ConfusionMatrix(task='multiclass',
                                                     num_classes=4,
                                                     normalize='true')

        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.val_cm = torchmetrics.ConfusionMatrix(task='multiclass',
                                                   num_classes=4,
                                                   normalize='true')

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.train_cm(y_hat, y)

        self.log('train/loss', loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log('train/acc_epoch', self.train_acc)
        fig, _ = self.train_cm.plot()
        self.logger.experiment.add_figure('train/cm_epoch', fig, self.global_step - 1)

        fig = vis(self.model, self.trainer.train_dataloader.dataset, self.device)
        self.logger.experiment.add_figure('train/vis', fig, self.global_step - 1)

        # log learning rate
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_cm(y_hat, y)

        self.log('val/loss', loss, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        self.log('val/acc_epoch', self.val_acc)
        fig, _ = self.val_cm.plot()
        self.logger.experiment.add_figure('val/cm_epoch', fig, self.global_step - 1)

        fig = vis(self.model, self.trainer.val_dataloaders.dataset, self.device)
        self.logger.experiment.add_figure('val/vis', fig, self.global_step - 1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]


def main(config):
    callbacks = [
        EarlyStopping(monitor='val/loss', patience=10, mode='min'),
    ]

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=config.exp_path,
        max_epochs=config.epoch,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        # deterministic=True,
        fast_dev_run=config.debug
    )

    if config.mode == 'train':
        train_transform = A.Compose([
            A.GaussianBlur(), A.GaussNoise(),
            A.RandomBrightnessContrast(), A.Flip(),
        ])

        train_dataset = ImageSeg(config.dataset_path, split='train',
                                transforms=train_transform)

        val_dataset = ImageSeg(config.dataset_path, split='val')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.workers
        )

        model = UNet()
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    args = parse_options()
    # L.seed_everything(args.seed)
    main(args)
