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


def process_image(row, data_dir, tile_2048_path, tile_2048_mask_path):
    tma = row["is_tma"]
    image_id = row["image_id"]
    img_path = data_dir / "train_images" / f"{row.image_id}.png"
    mask_path = row["mask_path"]

    img = pyvips.Image.new_from_file(img_path)
    mask = pyvips.Image.new_from_file(mask_path)

    height = img.height
    width = img.width

    if ~tma:
        # Iterate over the image in tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract a tile from the image and mask
                img_tile = img.crop(
                    x, y, min(tile_size, width - x), min(tile_size, height - y)
                ).numpy()

                gray = cv2.cvtColor(img_tile, cv2.COLOR_RGB2GRAY)
                _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                # black area ratio threshold start from here
                black_pixels = np.count_nonzero(binary_image == 0)
                total_pixels = np.prod(binary_image.shape[:2])
                black_area_ratio = black_pixels / total_pixels


                if black_area_ratio > 0.3:
                    # print(f"black_area_ratio: {black_area_ratio}")
                    continue

                mask_tile = mask.crop(
                    x, y, min(tile_size, width - x), min(tile_size, height - y)
                ).numpy()

                tile_save_path = tile_2048_path / f"{image_id}_tile_{x}_{y}.png"
                Image.fromarray(img_tile).resize(new_size, Image.LANCZOS).save(
                    tile_save_path
                )

                mask_tile_save_path = (
                    tile_2048_mask_path / f"{image_id}_tile_{x}_{y}.png"
                )

                Image.fromarray(mask_tile).resize(new_size, Image.LANCZOS).save(
                    mask_tile_save_path
                )
    else:
        pass


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

    tile_size = 2048
    white_thr = 240
    drop_thr = 0.6
    new_size = (512, 512)

    tile_2048_path = data_dir / "train_tiles_2048"
    shutil.rmtree(str(tile_2048_path), ignore_errors=True)
    tile_2048_path.mkdir(exist_ok=True)

    tile_2048_mask_path = data_dir / "train_tiles_2048_mask"
    shutil.rmtree(str(tile_2048_mask_path), ignore_errors=True)
    tile_2048_mask_path.mkdir(exist_ok=True)

    num_processes = 12
    Parallel(n_jobs=num_processes, backend="multiprocessing")(
        delayed(process_image)(row, data_dir, tile_2048_path, tile_2048_mask_path)
        for _, row in tqdm(train_df.iterrows(), total=len(train_df))
    )

    # for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    #     tma = row["is_tma"]
    #     image_id = row["image_id"]
    #     img_path = data_dir / "train_images" / f"{row.image_id}.png"
    #     mask_path = row["mask_path"]

    #     img = pyvips.Image.new_from_file(img_path)
    #     mask = pyvips.Image.new_from_file(mask_path)

    #     height = img.height
    #     width = img.width

    #     if ~tma:
    #         # Iterate over the image in tiles
    #         for y in range(0, height, tile_size):
    #             for x in range(0, width, tile_size):
    #                 # Extract a tile from the image and mask
    #                 img_tile = img.crop(
    #                     x, y, min(tile_size, width - x), min(tile_size, height - y)
    #                 ).numpy()

    #                 gray = cv2.cvtColor(img_tile, cv2.COLOR_BGR2GRAY)
    #                 _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    #                 # black area ratio threshold start from here
    #                 black_pixels = np.count_nonzero(binary_image == 0)
    #                 total_pixels = np.prod(binary_image.shape[:2])
    #                 black_area_ratio = black_pixels / total_pixels

    #                 # black_pixels = total_pixels - np.count_nonzero(binary_image)

    #                 # black_area_ratio = black_pixels / total_pixels

    #                 if black_area_ratio > 0.3:
    #                     print(f"black_area_ratio: {black_area_ratio}")
    #                     continue
    #                 # black area ratio threshold end here

    #                 # white area ratio threshold start from here
    #                 # white_pixels = np.count_nonzero(binary_image == 255)
    #                 # white_area_ratio = white_pixels / total_pixels

    #                 # if white_area_ratio > 0.7:
    #                 #     print(f"white_area_ratio: {white_area_ratio}")
    #                 #     continue
    #                 # white area ratio threshold end here

    #                 mask_tile = mask.crop(
    #                                 x, y, min(tile_size, width - x), min(tile_size, height - y)
    #                             ).numpy()

    #                 # mask_tile_tumor_mask =

    #                 tile_save_path = tile_2048_path / f"{image_id}_tile_{x}_{y}.png"
    #                 Image.fromarray(img_tile).resize(new_size, Image.LANCZOS).save(
    #                     tile_save_path
    #                 )

    #                 mask_tile_save_path = tile_2048_mask_path / f"{image_id}_tile_{x}_{y}.png"

    #                 Image.fromarray(mask_tile).resize(new_size, Image.LANCZOS).save(
    #                     mask_tile_save_path
    #                 )
    #     else:
    #         pass

    # break
