import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import shutil
import cv2
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path

import segmentation_models_pytorch as smp
from utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG:
    folds = 5
    # size of the image
    img_size = 512

    # batch_size and epochs
    batch_size = 16
    num_workers = 4
    seg_model_name = "efficientnet-b0"
    seg_model_weight = "exp-seg/seg-fold-0.pth"

    clc_model_name = "tf_efficientnetv2_s_in21ft1k"
    clc_model_weight = "exp-06/fold-0.pth"

    tile_size = 2048

    # target column
    target_cols = ["CC", "EC", "HGSC", "LGSC", "MC"]

    label_dict = {
        0: "CC",
        1: "EC",
        2: "HGSC",
        3: "LGSC",
        4: "MC"
    }

    num_classes = len(target_cols)


class UBCDataset(Dataset):
    def __init__(self, df, path_column, transforms=None):
        self.df = df
        self.image_ids = df["image_id"].values
        self.file_names = df[path_column].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_id = self.image_ids[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image_id": image_id,
            "image": img,
        }


def seg_infer(df):
    _image_ids = []
    _target_paths = []

    data_transforms = {
        "valid": A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    }

    valid_dataset = UBCDataset(test_df, path_column="thumbnails_file_paths", transforms=data_transforms["valid"])

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = smp.Unet(
        encoder_name=CFG.seg_model_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(CFG.seg_model_weight))
    print("seg weights loaded")

    for index, data in enumerate(valid_loader):
        images = data["image"].to(device)
        image_ids = data["image_id"]
        with torch.no_grad():
            outputs = model(images)

        probs = torch.sigmoid(outputs)
        probs = probs.detach().cpu().numpy()

        threshold = 0.50
        binary_masks = (probs > threshold).astype(int)

        for _index, image_id in enumerate(image_ids):
            _image_ids.append(str(image_id))
            target_path = f"dataset/infer_seg/{image_id}_seg.npy"
            _target_paths.append(target_path)
            np.save(target_path, binary_masks[_index][0])

    df["seg_paths"] = _target_paths

    return df


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class UBCModel(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=False, checkpoint_path=None):
        super(UBCModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)

    #         self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output


def infer_clc(tile_df, model):
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

    # test_dataset = UBCClcDataset(tile_df, transforms=get_transforms())
    # test_loader = DataLoader(test_dataset, batch_size=4,
    #                          num_workers=4, shuffle=False, pin_memory=True)
    # probs = []
    # for index, data in enumerate(test_loader):
    #     data = data["image"].to(device)
    #     outputs = model(data)
    #     p = torch.sigmoid(outputs)
    #     p = p.detach().cpu().numpy()
    #     probs.append(p)
    #
    #     del data
    #     del outputs
    #
    #     torch.cuda.empty_cache()
    # probs = np.concatenate(probs)
    # probs = np.mean(probs, 0)
    # max_index = np.argmax(probs)

    # label_dict = {
    #     0: "CC",
    #     1: "EC",
    #     2: "HGSC",
    #     3: "LGSC",
    #     4: "MC"
    # }
    #
    # max_prob = probs[max_index]
    #
    # if max_prob > 0.25:
    #     pred_class = label_dict[max_index]
    # else:
    #     pred_class = "Other"
    # return pred_class


def tile_image(df):
    tile_size = CFG.tile_size


    # model = UBCModel(CFG.clc_model_name)
    # model.to(device)
    # model.eval()
    # model.load_state_dict(torch.load(CFG.clc_model_weight))
    # print("clc weights loaded")


    rst_image_ids = []
    rst_labels = []
    for index, row in df.iterrows():

        # create tiles folder
        save_path = "tiles"
        os.makedirs(save_path, exist_ok=True)

        _img_ids = []
        _tile_file_paths = []

        image_id = row["image_id"]
        img_path = row["file_path"]
        mask_path = row["seg_paths"]
        is_tma = row["is_tma"]

        img = cv2.imread(str(img_path))

        height, width = img.shape[:2]
        mask = np.load(str(mask_path))
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        rows = height // tile_size
        cols = width // tile_size
        # print(is_tma)
        print(is_tma)
        if is_tma:
            print("is tma")
        else:
            for i in range(rows):
                for j in range(cols):
                    tile_mask = mask[
                        i * tile_size : (i + 1) * tile_size,
                        j * tile_size : (j + 1) * tile_size,
                    ]
                    true_percentage = tile_mask.sum() / tile_mask.size
                    if true_percentage < 0.5:
                        continue

                    tile = img[
                        i * tile_size : (i + 1) * tile_size,
                        j * tile_size : (j + 1) * tile_size,
                    ]

                    tile = cv2.resize(tile, (512, 512), interpolation=cv2.INTER_NEAREST)

                    tile_filename = f"{image_id}_{i}_{j}.png"
                    tile_path = f"{save_path}/{tile_filename}"
                    cv2.imwrite(tile_path, tile)
                    _img_ids.append(image_id)
                    _tile_file_paths.append(tile_path)
                    del tile
                    gc.collect()


        # del img
        # del mask
        # gc.collect()

        # tile_df = pd.DataFrame()
        # tile_df["image_id"] = _img_ids
        # tile_df["tile_file_paths"] = _tile_file_paths
        # print(tile_df)
        # pred_class = infer_clc(tile_df, model)
        # rst_image_ids.append(int(image_id))
        # rst_labels.append(pred_class)
        # shutil.rmtree(save_path)

if __name__ == '__main__':

    seed_everything()
    local = True

    if local:
        data_dir = Path("dataset")
        test_thumbnails_path = data_dir / "train_thumbnails"
        img_files = list(test_thumbnails_path.glob("*.png"))
        test_df = pd.DataFrame(img_files, columns=["thumbnails_file_paths"])
        test_df["image_id"] = test_df["thumbnails_file_paths"].apply(
            lambda x: str(x).split("/")[-1].split(".")[0].split("_")[0]
        )
        train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": "string"})
        test_df = test_df.merge(train_df, on="image_id", how="left")
        test_df["file_path"] = "dataset/train_images/" + test_df["image_id"].astype(str) + ".png"
    else:
        data_dir = Path("/kaggle/input/UBC-OCEAN")
        test_thumbnails_path = data_dir / "test_thumbnails"
        test_df = pd.read_csv(data_dir / "test.csv")
        test_df["is_tma"] = test_df["image_width"] < 6000
        test_df["file_path"] = "/kaggle/input/UBC-OCEAN/test_images/" + test_df["image_id"].astype(str) + ".png"
        test_df["thumbnails_file_paths"] = "/kaggle/input/UBC-OCEAN/test_thumbnails/" + test_df["image_id"].astype(
            str) + "_thumbnail.png"

    # print(test_df.head())
    # test_df = seg_infer(test_df)
    # test_df.to_csv("test_df.csv", index=False)

    test_df = pd.read_csv("test_df.csv")
    print(test_df.head())

    tile_image(test_df)