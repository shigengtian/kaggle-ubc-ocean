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


# def vips_read_image(image_path, longest_edge):
#     """
#     Read image using libvips

#     Parameters
#     ----------
#     image_path: str
#         Path of the image

#     Returns
#     -------
#     image: numpy.ndarray of shape (height, width, 3)
#         Image array
#     """

#     image_thumbnail = pyvips.Image.thumbnail(image_path, longest_edge)

#     return np.ndarray(
#         buffer=image_thumbnail.write_to_memory(),
#         dtype=np.uint8,
#         shape=[image_thumbnail.height, image_thumbnail.width, image_thumbnail.bands],
#     )


# def process_image(row, data_dir, target_path):
#     tma = row["is_tma"]
#     if not tma:
#         img_path = data_dir / "train_images" / f"{row.image_id}.png"
#         img = vips_read_image(str(img_path), longest_edge=5000)
#         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
#     else:
#         img_path = data_dir / "train_images" / f"{row.image_id}.png"
#         img = cv2.imread(str(img_path))
#         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)


def convert_rgb_to_labels(img_path: str, folder: str):
    name = os.path.basename(img_path)
    img = np.array(Image.open(img_path))
    # plt.imshow(img)
    bg = np.ones((img.shape[0], img.shape[1], 1)) * 128
    stack = np.concatenate((bg, img), axis=2)
    mask = np.argmax(stack, axis=2).astype(np.uint8)
    img_path = os.path.join(folder, name)
    Image.fromarray(mask).save(img_path)
    return img_path


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_images = sorted(glob(str(data_dir / "train_images/*.png")))
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    train_masks = sorted(glob(str(data_dir / "masks/*.png")))
    train_mask_df = pd.DataFrame({"mask_path": train_masks})
    train_mask_df["image_id"] = train_mask_df["mask_path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    train_df = train_mask_df.merge(train_df, on="image_id", how="left")

    # mask = 
