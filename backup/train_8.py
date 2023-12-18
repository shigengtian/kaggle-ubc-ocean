import os
import pandas as pd
import numpy as np
import random
import shutil
import cv2

import sklearn
import matplotlib.pyplot as plt
import yaml
import math
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
from sklearn.metrics import precision_score
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

    exp_name = "exp-09"

    model_name = "tf_efficientnetv2_s_in21ft1k"

    # seed for data-split, layer init, augs
    seed = 42

    # number of folds for data-split
    folds = 10

    # which folds to train
    selected_folds = [0, 1, 2, 3, 4]

    # size of the image
    img_size = 512

    batch_size = 8
    # batch_size and epochs
    epochs = 30
    num_workers = 8 

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

    s = 30.0
    m = 0.5
    ls_eps = 0.0
    easy_margin = False


# class UBCDataset(Dataset):
#     def __init__(self, df, transforms=None):
#         self.df = df
#         self.file_names = df["img_path"].values
#         self.labels = df["label"].values
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         img_path = self.file_names[index]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         label = self.labels[index]
#         # one_hot_label = np.zeros(CFG.num_classes)
#         # one_hot_label[label] = 1.0

#         if self.transforms:
#             img = self.transforms(image=img)["image"]

#         return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['img_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }
        

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
    return nn.CrossEntropyLoss()(outputs, labels)
    # return nn.BCEWithLogitsLoss()(outputs, labels)


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
            y_preds = model(images, labels)
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


@torch.inference_mode()
def valid_fn(valid_loader, model, epoch, criterion, fold):
    losses = AverageMeter()
    model.eval()
    bar = tqdm(total=len(valid_loader))
    bar.set_description(f"Valid Fold: {fold}, Epoch: {epoch + 1}")

    preds = []
    label = []
    
    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    
    for step, data in enumerate(valid_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)
        batch_size = labels.size(0)

        y_preds = model(images, labels)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        _, predicted = torch.max(model.softmax(y_preds), 1)
        
        # y_preds = torch.sigmoid(y_preds).detach().cpu().numpy()
        
        acc = torch.sum( predicted == labels )
        
        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        
        # preds.append(y_preds)


        # label.append(labels.to("cpu").numpy())

        torch.cuda.empty_cache()
        bar.update()

    # preds = np.concatenate(preds)
    # label = np.concatenate(label)

    # acc = np.sum(np.argmax(preds, axis=1) == np.argmax(label, axis=1))
    # score = acc / len(preds)
    score = epoch_acc
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

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class UBCModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False, checkpoint_path=None):
        super(UBCModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        embedding_size = 512
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.softmax = nn.Softmax(dim=1)
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, 
                                   CFG.num_classes,
                                   s=CFG.s, 
                                   m=CFG.m, 
                                   easy_margin=CFG.easy_margin, 
                                   ls_eps=CFG.ls_eps)

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding, labels)
        return output
    
    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding



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
            torch.save(
                model.state_dict(), f"{output_path}/{CFG.exp_name}-fold-{fold}.pth"
            )

    LOGGER.info(f"best acc: {best_acc:.4f}")
    LOGGER.info(f"========== fold: {fold} training end ==========")


def split_df(df):
    skf = StratifiedKFold(n_splits=CFG.folds)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
        df.loc[val_, "fold"] = int(fold)
    return df


if __name__ == "__main__":
    seed_everything()
    output_path = f"{CFG.exp_name}"
    os.makedirs(output_path, exist_ok=True)
    LOGGER = get_logger(f"{output_path}/train")

    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": "string"})

    def get_img_path(x):
        return f"{data_dir}/train_thumbnails_custom/{x}.png"

    train_df["img_path"] = train_df["image_id"].apply(get_img_path)
    train_df["label"] = train_df["label"].map(CFG.label_dict)

    train_df = split_df(train_df)

    for fold in CFG.selected_folds:

        LOGGER.info(f"Fold: {fold}")
        train_loop(train_df, fold)
        # break
