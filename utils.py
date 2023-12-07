import random
import os
import torch
import time
import math
import pandas as pd
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f"{int(m)}m {int(s)}s"


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{str(asMinutes(s))} (remain {str(asMinutes(rs))})"


def get_thumbnails_df(thumbnails_dir):
    """
    Get train_thumbnails_files_df
    """
    thumbnails_files = list(thumbnails_dir.glob("*.png"))
    print(f"thumbnails_files: {len(thumbnails_files)}")

    thumbnails_files_df = pd.DataFrame(
        thumbnails_files, columns=["thumbnails_file_paths"]
    )

    thumbnails_files_df["image_id"] = thumbnails_files_df[
        "thumbnails_file_paths"
    ].apply(lambda x: str(x).split("/")[-1].split("_")[0])
    return thumbnails_files_df


def get_mask_df(mask_dir):
    mask_files = list(mask_dir.glob("*.png"))
    print(f"mask_files: {len(mask_files)}")
    mask_files_df = pd.DataFrame(mask_files, columns=["mask_file_paths"])
    mask_files_df["image_id"] = mask_files_df["mask_file_paths"].apply(
        lambda x: str(x).split("/")[-1].split(".")[0]
    )

    return mask_files_df


def get_img_df(img_dir):
    img_files = list(img_dir.glob("*.png"))
    print(f"mask_files: {len(img_files)}")
    img_df = pd.DataFrame(img_files, columns=["img_file_paths"])

    img_df["image_id"] = img_df["img_file_paths"].apply(
        lambda x: str(x).split("/")[-1].split(".")[0]
    )

    return img_df
