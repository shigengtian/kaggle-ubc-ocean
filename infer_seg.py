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

    exp_name = "exp-seg"

    model_name = "efficientnet-b0"

    # seed for data-split, layer init, augs
    seed = 42

    # number of folds for data-split
    folds = 5

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


class CustomDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        self.img_path = df["train_thumbnails_file_paths"].values
        self.mask_path = df["segmentation_nps_file_paths"].values

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image = cv2.imread(str(self.img_path[index]))
        mask = np.load(str(self.mask_path[index])) / 255.0
        mask = mask[:, :, np.newaxis]

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        mask = np.transpose(mask, (2, 0, 1))
        return torch.tensor(image).float(), torch.tensor(mask).float()


def get_transforms():
    return A.Compose(
        [
            A.Resize(CFG.img_size, CFG.img_size),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            #     max_pixel_value=255.0,
            #     p=1.0,
            # ),
            # ToTensorV2(),
        ],
        p=1.0,
    )


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)

    valid_dataset = CustomDataset(valid_folds, transforms=get_transforms())

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
    model.load_state_dict(torch.load(f"{output_path}/fold-{fold}.pth"))
    print("weights loaded")

    for index, (images, label) in enumerate(valid_loader):
        images = images.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(images)

        probs = torch.sigmoid(outputs)
        probs = probs.detach().cpu().numpy()

        print(probs.shape)
        threshold = 0.5  # You can adjust this threshold based on your task
        binary_masks = (probs > threshold).astype(int)
        cv2.imwrite("pred.png", binary_masks[1][0] * 255)

        label = label.detach().cpu().numpy()
        cv2.imwrite("label.png", label[1][0] * 255)

        break


if __name__ == "__main__":
    seed_everything()
    output_path = f"{CFG.exp_name}"
    os.makedirs(output_path, exist_ok=True)
    LOGGER = get_logger(f"{output_path}/train")

    data_dir = Path("dataset")
    thumbnails_dir = Path(data_dir / "train_thumbnails")
    train_images = sorted(glob(str(thumbnails_dir / "*.png")))
    train_thumbnails_dir = data_dir / "train_thumbnails"

    segmentation_nps = sorted(glob(str(data_dir / "masks_np" / "*.npy")))

    segmentation_nps_df = pd.DataFrame(
        segmentation_nps, columns=["segmentation_nps_file_paths"]
    )

    segmentation_nps_df["image_id"] = segmentation_nps_df[
        "segmentation_nps_file_paths"
    ].apply(lambda x: str(x).split("/")[-1].split(".")[0])

    print("segmentation_nps_df")
    print(segmentation_nps_df.head())

    train_thumbnails_files = list(train_thumbnails_dir.glob("*.png"))
    print(f"train_thumbnails_files: {len(train_thumbnails_files)}")

    train_thumbnails_files_df = pd.DataFrame(
        train_thumbnails_files, columns=["train_thumbnails_file_paths"]
    )

    train_thumbnails_files_df["image_id"] = train_thumbnails_files_df[
        "train_thumbnails_file_paths"
    ].apply(lambda x: str(x).split("/")[-1].split("_")[0])

    print("train_thumbnails_files_df")
    print(train_thumbnails_files_df.head())

    train_seg = segmentation_nps_df.merge(
        train_thumbnails_files_df, on="image_id", how="left"
    )

    print(train_seg.head())

    kfold = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_seg)):
        train_seg.loc[valid_idx, "fold"] = int(fold)

    print(train_seg.head())

    for fold in CFG.selected_folds:
        LOGGER.info(f"Fold: {fold}")
        train_loop(train_seg, fold)
        break
