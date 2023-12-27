import os
import gc
import cv2
import glob
import zipfile
import time
import threading
import pyvips
import random as rn
from tqdm import tqdm
from PIL import Image

# from threading import Thread
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from joblib import Parallel, delayed
from PIL import Image
import shutil


from PIL import Image


def center_crop(image_id, target_size=2000):
    # wsi_name = tma_path.split('/')[-1].split('.')[0]

    img_path = str(data_dir/"train_images"/f"{image_id}.png")
    
    tma_image = Image.open(img_path)
    tma_width, tma_height = tma_image.size

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    left = (tma_width - target_size[0]) // 2
    upper = (tma_height - target_size[1]) // 2
    right = left + target_size[0]
    lower = upper + target_size[1]

    cropped_image = tma_image.crop((left, upper, right, lower))

    target_img_path = target_path / f"{row.image_id}.png"

    cropped_image.save(str(target_img_path))


if __name__ == "__main__":
    data_dir = Path("dataset")

    train_df = pd.read_csv(data_dir / "train.csv")
    tm_train_df = train_df[train_df["is_tma"] == True]
    target_path = data_dir / "tma_center_crop"
    target_path.mkdir(exist_ok=True)

    for index, row in tm_train_df.iterrows():
        center_crop(row.image_id)

