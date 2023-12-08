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

    exp_name = "exp-04"

    model_name = "tf_efficientnetv2_s_in21ft1k"

    # seed for data-split, layer init, augs
    seed = 42

    # number of folds for data-split
    folds = 10

    # which folds to train
    selected_folds = [0, 1, 2, 3, 4]

    # size of the image
    img_size = 512

    # batch_size and epochs
    batch_size = 8
    epochs = 30
    num_workers = 20

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


class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["label"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        label = np.eye(CFG.num_classes)[label]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "label": torch.tensor(label, dtype=torch.float32)}


def get_transforms(data):
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.ShiftScaleRotate(p=0.75),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ],
                    p=0.4,
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.CoarseDropout(
                    max_holes=1,
                    max_width=int(512 * 0.3),
                    max_height=int(512 * 0.3),
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
            is_check_shapes=False,
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


def criterion(outputs, labels):
    return nn.BCEWithLogitsLoss()(outputs, labels)


def train_fn(train_loader, model, optimizer, epoch, scheduler, criterion, fold):
    losses = AverageMeter()
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    bar = tqdm(total=len(train_loader))
    bar.set_description(f"TRAIN Fold: {fold}, Epoch: {epoch + 1}")

    for step, data in enumerate(train_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)
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


def valid_fn(valid_loader, model, epoch, criterion, fold):
    losses = AverageMeter()
    model.eval()
    bar = tqdm(total=len(valid_loader))
    bar.set_description(f"Valid Fold: {fold}, Epoch: {epoch + 1}")

    preds = []
    label = []
    for step, data in enumerate(valid_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)
        batch_size = labels.size(0)

        y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        y_preds = torch.sigmoid(y_preds).detach().cpu().numpy()
        preds.append(y_preds)

        label.append(labels.to("cpu").numpy())

        torch.cuda.empty_cache()
        bar.update()

    preds = np.concatenate(preds)
    label = np.concatenate(label)

    score = balanced_accuracy_score(label.argmax(axis=1), preds.argmax(axis=1))
    return losses.avg, score


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class UBCModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False, checkpoint_path=None):
        super(UBCModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    train_dataset = UBCDataset(train_folds, transforms=get_transforms("train"))
    valid_dataset = UBCDataset(valid_folds, transforms=get_transforms("valid"))

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

    model = UBCModel(CFG.model_name, CFG.num_classes, pretrained=True)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=1e-6, T_max=500
    )

    best_acc = -1.0
    best_loss = np.inf
    for epoch in range(CFG.epochs):
        train_loss = train_fn(
            train_loader, model, optimizer, epoch, scheduler, criterion, fold
        )

        valid_loss, valid_acc = valid_fn(valid_loader, model, epoch, criterion, fold)

        LOGGER.info(
            f"Epoch {epoch + 1} | Valid Loss: {valid_loss:.4f} | acc:{valid_acc:.4f}"
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            LOGGER.info(f"Epoch {epoch + 1} | Best Valid acc: {best_acc:.4f}")
            torch.save(model.state_dict(), f"{output_path}/fold-{fold}.pth")


    LOGGER.info(f"best acc: {best_acc:.4f}")
    LOGGER.info(f"========== fold: {fold} training end ==========")


if __name__ == "__main__":
    seed_everything()
    output_path = f"{CFG.exp_name}"
    os.makedirs(output_path, exist_ok=True)
    LOGGER = get_logger(f"{output_path}/train")

    data_dir = Path("dataset")
    resized_dir = Path(data_dir / "train_images_512")
    train_images = sorted(glob(str(resized_dir / "*.png")))

    def get_train_file_path(image_id):
        return str(resized_dir / f"{image_id}.png")

    train_df = pd.read_csv(data_dir / "train.csv")
    train_df["label"] = train_df["label"].map(CFG.label_dict)
    

    train_df["file_path"] = train_df["image_id"].apply(get_train_file_path)
    not_tma_df = train_df[train_df["is_tma"] == False].reset_index(drop=True) 
    print(not_tma_df)

    skf = StratifiedKFold(n_splits=CFG.folds)

    for fold, (_, val_) in enumerate(skf.split(X=not_tma_df, y=not_tma_df.label)):
        not_tma_df.loc[val_, "fold"] = int(fold)

    for fold in CFG.selected_folds:
        LOGGER.info(f"Fold: {fold}")
        train_loop(not_tma_df, fold)
        break
