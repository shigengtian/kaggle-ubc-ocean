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
    model_name = "tf_efficientnetv2_s_in21ft1k"

    # seed for data-split, layer init, augs
    seed = 42

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

    target_cols = ["CC", "EC", "HGSC", "LGSC", "MC"]

    label_dict = {0: "CC", 1: "EC", 2: "HGSC", 3: "LGSC", 4: "MC"}

    num_classes = len(target_cols)


class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img}


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
        p=1.0,
    )


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


def infer_clc(test_df):
    valid_dataset = UBCDataset(test_df, transforms=get_transforms())

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = UBCModel(CFG.model_name, CFG.num_classes, pretrained=False)
    model.load_state_dict(torch.load("exp-04/fold-0.pth"))
    model.to(device)
    model.eval()
    preds = []

    for step, data in enumerate(valid_loader):
        images = data["image"].to(device)
        y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).detach().cpu().numpy()
        preds.append(y_preds)
        torch.cuda.empty_cache()

    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)

    test_df["label"] = preds
    test_df["label"] = test_df["label"].map(CFG.label_dict)
    
    print(test_df.head())

if __name__ == "__main__":
    seed_everything()

    data_dir = Path("dataset")
    resized_dir = Path(data_dir / "train_images_512")
    train_images = sorted(glob(str(resized_dir / "*.png")))
    train_images = train_images[:10]
    print(train_images[:10])

    test_df = pd.DataFrame()
    test_df["file_path"] = train_images
    print(test_df.head())

    infer_clc(test_df)