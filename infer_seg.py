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
    batch_size = 32
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
        self.img_paths = df["tile_path"].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {"image": img,}


def get_transforms():
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
        p=1.0,)
    
    


if __name__ == "__main__":
    seed_everything()
    # output_path = f"{CFG.exp_name}"
    # os.makedirs(output_path, exist_ok=True)
    # LOGGER = get_logger(f"{output_path}/train")

    data_dir = Path("dataset")
    train_tiles = sorted(glob(str(data_dir / "train_tiles" / "*.png")))
    tile_df = pd.DataFrame(train_tiles, columns=["tile_path"])

    print(tile_df.head())
    # tile_df = tile_df[:10] 
    model = smp.Unet(
        encoder_name=CFG.model_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(f"exp-seg/seg-fold-0.pth"))
    model.to(device)
    
    
    test_dataset = UBCDataset(tile_df, transforms=get_transforms())
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    bar = tqdm(total=len(test_dataset_loader))
    bar.set_description(f"Inferencing")
    
    mask_ratio = []
    for index, data in enumerate(test_dataset_loader):
        images = data["image"].to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            outputs = model(images)
            
        probs = torch.sigmoid(outputs)
        probs = probs.detach().cpu().numpy()

        threshold = 0.33
        binary_masks = (probs > threshold).astype(int)
        for _index in range(batch_size):
            true_pixel_ratio = np.count_nonzero(binary_masks[_index]) /(512*512)
            mask_ratio.append(true_pixel_ratio)
        bar.update()
        
    print("mask ratio")
    print(mask_ratio)
    tile_df["mask_ratio"] = mask_ratio
    tile_df.to_csv("dataset/tile_df.csv", index=False)