import os
import pandas as pd
import numpy as np
import random
import shutil
import cv2

import sklearn
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms
import timm

import joblib
from pathlib import Path
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import seed_everything, AverageMeter, get_logger

import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG:
    wandb = True
    competition = "ubc-ocean"
    _wandb_kernel = "shigengtian/ubc-ocean"

    exp_name = "exp-seg-03"

    model_name = "efficientnet-b0"

    # seed for data-split, layer init, augs
    seed = 42

    # number of folds for data-split
    folds = 10

    # which folds to train
    selected_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # size of the image
    img_size = 512

    # batch_size and epochs
    batch_size = 8
    epochs = 10
    num_workers = 12

    lr = 1e-4
    weight_decay = 1e-6
    # target column
    target_cols = ["CC", "EC", "HGSC", "LGSC", "MC"]

    label_dict = {
        "CC": 0,
        "EC": 1,
        "HGSC": 2,
        "LGSC": 3,
        "MC": 4,
    }

    num_classes = len(target_cols)


class CustomDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        self.img_path = df["tile_path"].values
        self.mask_path = df["tile_mask_path"].values

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image = cv2.imread(str(self.img_path[index]))
        mask = cv2.imread(str(self.mask_path[index]))

        mask = mask[..., 2]
        mask = (mask > 0).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0)
        return image, mask


def get_transforms(data):
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ],
                    p=0.4,
                ),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                # A.CoarseDropout(
                #     max_holes=1,
                #     max_width=int(512 * 0.3),
                #     max_height=int(512 * 0.3),
                #     mask_fill_value=0,
                #     p=0.5,
                # ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
            is_check_shapes=True,
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        )


def train_fn(train_loader, model, optimizer, epoch, scheduler, fold):
    losses = AverageMeter()
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    bar = tqdm(total=len(train_loader))
    bar.set_description(f"TRAIN Fold: {fold}, Epoch: {epoch + 1}")

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.empty_cache()
        bar.update()

    return losses.avg


def valid_fn(valid_loader, model, epoch, fold):
    losses = AverageMeter()
    model.eval()
    bar = tqdm(total=len(valid_loader))
    bar.set_description(f"Valid Fold: {fold}, Epoch: {epoch + 1}")

    preds = []
    label = []
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        torch.cuda.empty_cache()
        bar.update()

    return losses.avg


JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    train_dataset = CustomDataset(train_folds, transforms=get_transforms("train"))
    valid_dataset = CustomDataset(valid_folds, transforms=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = smp.Unet(
        encoder_name=CFG.model_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=1e-6, T_max=500
    )

    best_loss = np.inf
    for epoch in range(CFG.epochs):
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, fold)

        valid_loss = valid_fn(valid_loader, model, epoch, fold)

        LOGGER.info(f"Epoch {epoch + 1} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            LOGGER.info(f"Epoch {epoch + 1} | Best Valid Loss: {best_loss:.4f}")
            torch.save(model.state_dict(), f"{output_path}/seg-03-fold-{fold}.pth")

    LOGGER.info(f"best loss: {best_loss:.4f}")
    LOGGER.info(f"========== fold: {fold} training end ==========")


if __name__ == "__main__":
    seed_everything()
    output_path = f"{CFG.exp_name}"
    os.makedirs(output_path, exist_ok=True)
    LOGGER = get_logger(f"{output_path}/train")

    if False:
        data_dir = Path("dataset")
        tile_path = data_dir / "train_tiles_2048"
        tile_mask_path = data_dir / "train_tiles_2048_mask"

        tile_files = sorted(glob(str(tile_path / "*.png")))
        tile_df = pd.DataFrame()
        tile_df["tile_path"] = tile_files
        tile_df["tile_id"] = tile_df["tile_path"].apply(lambda x: x.split("/")[-1])

        tile_mask_df = pd.DataFrame()
        tile_mask_files = sorted(glob(str(tile_mask_path / "*.png")))
        tile_mask_df["tile_mask_path"] = tile_mask_files
        tile_mask_df["tile_id"] = tile_mask_df["tile_mask_path"].apply(
            lambda x: x.split("/")[-1]
        )
        print(tile_mask_df.head())

        train_df = tile_df.merge(tile_mask_df, on="tile_id", how="left")
        print(train_df.head())

        tumor_ratio = []
        for index, row in train_df.iterrows():
            mask = cv2.imread(row.tile_mask_path)

            red_channel = mask[..., 2]

            tummor_area_ratio = np.sum(red_channel > 0) / red_channel.size
            tumor_ratio.append(tummor_area_ratio)

        train_df["tumor_ratio"] = tumor_ratio
        train_df.to_csv("dataset/train_tile_mask.csv", index=False)

    train_df = pd.read_csv("dataset/train_tile_mask.csv")
    train_df = train_df[train_df["tumor_ratio"] > 0.8].reset_index(drop=True)
    print(train_df.head())

    kfold = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        train_df.loc[valid_idx, "fold"] = int(fold)

    print(train_df.head())

    for fold in CFG.selected_folds:
        LOGGER.info(f"Fold: {fold}")
        train_loop(train_df, fold)
        # break
